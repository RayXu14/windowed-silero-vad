import asyncio
import json
import numpy as np
import websockets
import os
import time
import torch
from scipy.io import wavfile

class SimpleVADClient:
    """简化的VAD WebSocket客户端"""
    
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.websocket = None
        self.server_ready = False
        self.server_ready_event = asyncio.Event()
        self.last_message_time = None
        self.receive_task = None
    
    async def connect(self):
        """连接WebSocket并等待服务器就绪"""
        print("连接到服务器...")
        self.websocket = await websockets.connect(self.uri)
        print("WebSocket连接已建立")
        
        # 启动接收循环
        self.receive_task = asyncio.create_task(self._receive_loop())
        print("接收循环已启动")
        
        # 等待服务器就绪
        print("等待服务器加载VAD模型...")
        await self.server_ready_event.wait()
        print("服务器就绪，可以开始发送数据")
    
    async def _receive_loop(self):
        """接收VAD结果和状态消息"""
        print("接收循环开始...")
        try:
            while True:
                try:
                    message = await self.websocket.recv()
        
                    
                    result = json.loads(message)
                    
                    if result.get("type") == "status":
                        status = result.get("status")
                        message_text = result.get("message", "")
                        print(f"服务器状态: {status} - {message_text}")
                        
                        if status == "ready":
                            print("设置服务器就绪状态")
                            self.server_ready = True
                            self.server_ready_event.set()
                            
                    elif result.get("type") == "vad":
                        # 输出VAD检测日志
                        print("检测到语音活动")
                        self.last_message_time = time.time()
                            
                    elif result.get("type") == "asr":
                        print(f"语音转文本结果: {result.get('text')}")
                        self.last_message_time = time.time()
                        
                    elif result.get("type") == "error":
                        print(f"服务器错误: {result.get('error')}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket连接已关闭")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
                    
        except Exception as e:
            print(f"接收消息时出错: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_audio(self, audio_data, timestamp):
        """发送音频数据"""
        message = {
            "type": "audio_chunk",
            "data": audio_data.tolist(),
            "timestamp": timestamp
        }
        await self.websocket.send(json.dumps(message))
    
    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()

async def send_audio_chunks(client, combined_wav, chunk_size, send_start_time):
    """发送音频块的异步函数"""
    chunk_count = 0
    for i in range(0, len(combined_wav), chunk_size):
        chunk = combined_wav[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        chunk_start = time.time()
        
        # 计算音频块的时间戳：初始时间 + 音频块采样点偏移 / 采样率
        chunk_timestamp = send_start_time + (i / 16000)
        await client.send_audio(chunk, chunk_timestamp)
        chunk_end = time.time()
        chunk_count += 1
        if chunk_count % 1000 == 0:  # 只打印前5个和每100个
            await asyncio.sleep(0.01)  # 异步证明
            print(f'发送第{chunk_count}块耗时: {(chunk_end - chunk_start)*1000:.2f} ms')
    
    # 发送100块空音频确保服务端处理完成
    print("发送100块空音频以确保服务端处理完成...")
    empty_chunk = np.zeros(chunk_size, dtype=np.float32)
    for i in range(100):
        # 计算空音频块的时间戳
        chunk_timestamp = send_start_time + (len(combined_wav) + i * chunk_size) / 16000
        await client.send_audio(empty_chunk, chunk_timestamp)
        chunk_count += 1
    
    send_done_time = time.time()
    print(f'发送完成，共{chunk_count}块（包含100块空音频），发送耗时: {(send_done_time - send_start_time):.2f} 秒')
    return send_done_time

async def get_vad_results(wav_path, chunk_size=512):
    """获取音频文件的VAD结果列表"""
    client = SimpleVADClient()
    
    try:
        await client.connect()
        
        # 读取并连接所有音频文件
        wav_files = sorted([f for f in os.listdir(wav_path) 
                           if f.endswith('.wav') and not f.startswith('.')])
        
        wavs = []
        for f in wav_files:
            # 直接读取wav文件
            sample_rate, audio_data = wavfile.read(os.path.join(wav_path, f))
            
            # 转换为float32并归一化
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('不被支持的数据类型')
            
            # 如果需要重采样到16000Hz
            if sample_rate != 16000:
                print(f"警告: {f} 的采样率是 {sample_rate}Hz，不是期望的16000Hz")
            
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(audio_data)
            wavs.append(audio_tensor)
        combined_wav = torch.concat(wavs)
        
        # 服务器就绪后开始计时
        print("开始发送音频数据...")
        send_start_time = time.time()
        
        # 创建发送任务，让发送和接收真正并发执行
        send_task = asyncio.create_task(
            send_audio_chunks(client, combined_wav, chunk_size, send_start_time)
        )
        
        # 不等待发送完成，让发送和接收真正并发
        # 等待发送完成
        send_done_time = await send_task
        
        # 等待处理完成 - 等待1秒让服务器处理完所有音频
        print("等待服务器处理完成...")
        await asyncio.sleep(1.0)
        
        # 计算处理时间
        total_send_time = send_done_time - send_start_time
        print(f'音频发送耗时: {total_send_time:.2f} 秒')
        
        if client.last_message_time:
            last_vad_time = client.last_message_time - send_start_time
            print(f'最后VAD结果时间: {last_vad_time:.2f} 秒')
        
    finally:
        await client.close()

if __name__ == "__main__":
    wav_path = "exp_data/2025-07-29_16-15-16-756/equidistant_audio_records"
    
    print("开始VAD处理...")
    start_time = time.time()
    
    asyncio.run(get_vad_results(wav_path, chunk_size=320))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'\n总处理时间: {total_time:.2f} 秒')
