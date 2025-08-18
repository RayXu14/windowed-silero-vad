from collections import deque
from enum import Enum
import json
import numpy as np
import torch
import time
import os
import torchaudio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad
import uvicorn
from loguru import logger
from asr import asr, format_str_v3

# 配置 loguru 日志 - 最简单配置：控制台 + 文件
logger.add("vad_server.log")

app = FastAPI(title="VAD WebSocket Server", version="1.0.0")

# 应用配置
class AppConfig:
    prob_threshold = None
    smoothing_window = None
    required_hits = None
    required_misses = None
    prebuffer = None
    lang = None

config = AppConfig()

class State(Enum):
    IDLE = 0
    ACTIVE = 1

class VADProcessor:
    """VAD处理器类，封装VAD计算过程"""
    
    # 类变量 - VAD配置常量
    SAMPLING_RATE = 16000
    DATA_TYPE = np.float32
    CHANNELS = 1  # 单声道
    WINDOW_SIZE = 512 if SAMPLING_RATE == 16000 else 256  # 32ms * 16000 / 1000
    
    def __init__(self, smoothing_window: int, prob_threshold: float, required_hits: int, required_misses: int, prebuffer: int, lang: str):
        # 加载VAD模型
        self.prob_model = load_silero_vad(onnx=True)
        self.cache_asr = {}
        self.lang = lang

        # 加载初始状态
        self.state = State.IDLE
        
        # 音频缓冲区
        self.audio_buffer = np.array([], dtype=self.DATA_TYPE)

        # 移动平均窗口和阈值
        self.prob_window = deque(maxlen=smoothing_window)
        self.prob_threshold = prob_threshold
        
        # 状态转换参数
        self.required_hits = required_hits
        self.required_misses = required_misses
        
        # 预缓冲参数和循环缓冲区
        self.prebuffer = prebuffer
        self.prebuffer_queue = deque(maxlen=required_hits + prebuffer)
        
        # 状态转换计数器
        self.hit_count = 0
        self.miss_count = 0
        
        # 语音段缓冲：收集完整语音段的音频数据
        self.speech_segment_buffer = []
        
        # 保存目录和计数器
        self.save_dir = "saved_audio_segments"
        os.makedirs(self.save_dir, exist_ok=True)
        self.segment_counter = 0
        

    def get_smooth_values(self, prob):
        self.prob_window.append(prob)
        smoothed_prob = np.mean(self.prob_window)
        return smoothed_prob
    
    def transcrib_and_save_speech_segment(self):
        """保存完整的语音段"""
        if not self.speech_segment_buffer:
            logger.warning("No speech segment to save")
            return None, None, None
        
        try:
            # 合并所有音频tensor
            audio_tensors = [chunk["audio_tensor"] for chunk in self.speech_segment_buffer]
            merged_audio = torch.cat(audio_tensors, dim=0)
            asr_result = asr(merged_audio, self.lang, self.cache_asr, use_itn=True)
            text = format_str_v3(asr_result[0]['text'])
            
            start_timestamp = self.speech_segment_buffer[-1]["timestamp"]
            segment_datetime = datetime.fromtimestamp(start_timestamp)
            timestamp_str = segment_datetime.strftime("%Y%m%d_%H%M%S")
            
            self.segment_counter += 1
            filename = f"segment_{timestamp_str}_{self.segment_counter:03d}.wav"
            filepath = os.path.join(self.save_dir, filename)
            
            # 使用torchaudio保存音频文件
            # 需要添加batch维度 (1, samples)
            audio_to_save = merged_audio.unsqueeze(0)
            torchaudio.save(filepath, audio_to_save, self.SAMPLING_RATE)
            
            duration = len(merged_audio) / self.SAMPLING_RATE
            
            logger.info(f"Saved speech segment: {filename} (duration: {duration:.2f}s)")
            return text, json.dumps(asr_result[0], ensure_ascii=False), start_timestamp
        except Exception as e:
            logger.opt(exception=True).error("Failed to save speech segment")
            # 即使保存失败也要清空缓冲区
            return None, None, None
        finally:
            self.speech_segment_buffer.clear()
    
    def process_audio_chunk(self, audio_data, timestamp):
        """
        处理音频块，使用生成器逐个返回VAD结果
        
        Args:
            audio_data: 音频数据列表
            timestamp: 音频块的时间戳（必须提供）
            
        Yields:
            dict: VAD结果字典
        """
        if not isinstance(audio_data, list):
            return
        
        if timestamp is None:
            raise ValueError("客户端必须提供音频块时间戳！")
        
        # 将音频数据转换为numpy数组并添加到缓冲区
        new_audio = np.array(audio_data, dtype=self.DATA_TYPE)
        # print(self.audio_buffer.shape, new_audio.shape)
        self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
        
        # 处理缓冲区中的完整音频块
        while len(self.audio_buffer) >= self.WINDOW_SIZE:
            # 提取一个完整的音频块
            audio_chunk = self.audio_buffer[:self.WINDOW_SIZE]
            self.audio_buffer = self.audio_buffer[self.WINDOW_SIZE:]
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(audio_chunk)
            
            # 使用prob_model计算语音概率
            speech_prob = self.prob_model(audio_tensor, self.SAMPLING_RATE).item()
            smoothed_prob = self.get_smooth_values(speech_prob)
            
            # 状态管理逻辑
            is_hit = smoothed_prob > self.prob_threshold
            
            # 创建当前音频块的信息（包含torch tensor和时间戳）
            current_chunk = {
                "audio_tensor": audio_tensor,
                "timestamp": timestamp  # 比较粗糙的时间戳
            }
            
            # 将当前音频块添加到预缓冲循环队列中（无论是否hit都要保存）
            self.prebuffer_queue.append(current_chunk)
            
            if is_hit:
                self.hit_count += 1
                self.miss_count = 0  # 重置miss计数器
                
                # 在IDLE状态下hit
                if self.state == State.IDLE:
                    
                    # 检查是否达到转换条件
                    if self.hit_count >= self.required_hits:
                        self.state = State.ACTIVE
                        
                        # 获取语音段开始时间戳（使用第一个命中的音频块时间戳）
                        speech_start_timestamp = self.prebuffer_queue[-self.hit_count]['timestamp']

                        yield {
                            "type": "vad",
                            'timestamp': speech_start_timestamp
                        }
                        
                        while self.prebuffer_queue:
                            self.speech_segment_buffer.append(self.prebuffer_queue.popleft())
                        
                # 在ACTIVE状态下hit
                elif self.state == State.ACTIVE:
                
                    yield {
                        "type": "vad",
                        'timestamp': current_chunk['timestamp']
                    }

                    while self.prebuffer_queue:
                        self.speech_segment_buffer.append(self.prebuffer_queue.popleft())
                        
            else:
                self.miss_count += 1
                self.hit_count = 0  # 重置hit计数器
                
                # 在IDLE状态下miss，不做任何处理
                if self.state == State.IDLE:
                    pass
                    
                # 在ACTIVE状态下miss
                elif self.state == State.ACTIVE:
                
                    yield {
                        "type": "vad",
                        'timestamp': current_chunk['timestamp']
                    }
                    # 检查是否达到转换条件
                    if self.miss_count >= self.required_misses:
                        self.state = State.IDLE
                        # 保存完整的语音段
                        text, full_info_str, asr_timestamp = self.transcrib_and_save_speech_segment()
                        if text is None:
                            logger.error('未能生成文本')
                            yield {
                                'type': 'error',
                                'error': '未能生成文本，请查看服务器端状况'
                            }
                        else:
                            yield {
                                'type': 'asr',
                                'text': text,
                                'full_info_str': full_info_str,
                                'timestamp': asr_timestamp
                            }
                            logger.info(f'生成文本：{text}')

# 获取服务器信息的统一函数
def get_server_info():
    return {
        "message": "VAD WebSocket 服务器 (异步音频流) 正在运行",
        "websocket_endpoint": "/ws",
        "audio_specs": {
            "sampling_rate": f"{VADProcessor.SAMPLING_RATE} Hz",
            "channels": VADProcessor.CHANNELS,
            "data_type": VADProcessor.DATA_TYPE.__name__,
            "chunk_size": f"{VADProcessor.WINDOW_SIZE} 样本"
        },
        "vad_config": vars(config),
        "usage": "连接到 WebSocket 端点发送音频数据进行 VAD 检测"
    }


@app.get("/")
async def root():
    return get_server_info()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted - NEW VERSION")
    
    # 为每个连接创建独立的 VAD 处理器，使用配置的参数
    logger.info(f"Creating VAD processor for new connection...")
    vad_processor = VADProcessor(
        smoothing_window=config.smoothing_window, 
        prob_threshold=config.prob_threshold,
        required_hits=config.required_hits,
        required_misses=config.required_misses,
        prebuffer=config.prebuffer,
        lang=config.lang,
    )
    logger.info("WebSocket connection established with dedicated VAD processor")
    
    # 发送就绪状态
    logger.info("Sending ready status...")
    await websocket.send_json({
        "type": "status",
        "status": "ready", 
        "message": "VAD 和 ASR 模型加载完成，可以开始发送音频数据"
    })
    logger.info("Ready status sent")
    
    try:
        while True:
            message = await websocket.receive_json()
            
            if message.get("type") == "audio_chunk":
                # 获取音频数据和时间戳
                audio_data = message.get("data")
                timestamp = message.get("timestamp")
                
                if audio_data is None:
                    continue
                
                # 使用VAD处理器处理音频数据
                vad_results = vad_processor.process_audio_chunk(audio_data, timestamp)
                
                # 发送所有VAD结果
                for result in vad_results:
                    await websocket.send_json(result)
            
            else:
                # 对不支持的消息类型发送错误回应
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {message.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except json.JSONDecodeError as e:
        # JSON解析错误应该报告给客户端
        logger.opt(exception=True).error("JSON decode error")
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Invalid JSON format: {str(e)}"
            })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected while sending JSON error")
    except Exception as e:
        logger.opt(exception=True).error("Error processing message")
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Processing error: {str(e)}"
            })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected while sending processing error")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VAD WebSocket 服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='启用自动重载')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], 
                       help='日志级别')
    parser.add_argument('--prob-threshold', type=float, default=0.8, 
                       help='移动平均概率阈值，只有大于此值才返回结果')
    parser.add_argument('--smoothing-window', type=int, default=1, 
                       help='移动平均窗口大小')
    parser.add_argument('--required-hits', type=int, default=5, 
                       help='从IDLE进入ACTIVE状态需要的连续命中次数')
    parser.add_argument('--required-misses', type=int, default=16, 
                       help='从ACTIVE进入IDLE状态需要的连续未命中次数')
    parser.add_argument('--prebuffer', type=int, default=10, 
                       help='从IDLE转换到ACTIVE时包含的之前音频块数量')
    parser.add_argument('--lang', type=str, default='auto', 
                       help='ASR语言')
    
    args = parser.parse_args()

    assert args.prebuffer + args.required_hits <= args.required_misses
    
    # 设置全局配置
    config.prob_threshold = args.prob_threshold
    config.smoothing_window = args.smoothing_window
    config.required_hits = args.required_hits
    config.required_misses = args.required_misses
    config.prebuffer = args.prebuffer
    config.lang = args.lang
    
    # 直接输出服务器信息
    server_info = get_server_info()
    logger.info("启动服务器...")
    logger.info(f"地址: http://{args.host}:{args.port}")
    logger.info(f"WebSocket 端点: ws://{args.host}:{args.port}{server_info['websocket_endpoint']}")
    logger.info(json.dumps(server_info, ensure_ascii=False, indent=2))
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("-" * 50)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
