import asyncio
import base64
import io
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
import httpx
import uvicorn
from loguru import logger
from speaker_verification import EmbeddingManager, SystemConfig

app = FastAPI(title="VAD WebSocket Server", version="1.0.0")

# 应用配置
class AppConfig:
    prob_threshold = None
    exit_prob_threshold = None
    smoothing_window = None
    smoothing_method = None
    required_hits = None
    required_misses = None
    prebuffer = None
    lang = None
    volume_threshold = None
    snr_margin_db = None
    active_snr_margin_db = None
    noise_floor_window = None
    noise_update_prob_threshold = None
    enable_speaker_verification = None
    asr_url = None
    vad_silence_threshold = None
    # early-onset 辅助回溯：用原始(未平滑)概率把语音段左边界往前精修
    enable_early_onset = None
    early_onset_floor = None
    early_onset_release = None
    early_onset_max_lookback = None
    # tail-extend 段尾延伸：用原始概率把语音段右边界往后延伸，找回被切掉的尾音(镜像 early-onset)
    enable_tail_extend = None
    tail_floor = None
    tail_release = None
    tail_max_lookahead = None


config = AppConfig()


class State(Enum):
    IDLE = 0
    ACTIVE = 1


def calculate_db(chunk):
    """计算音频块的分贝值"""
    rms = np.sqrt(np.mean(np.square(chunk)))
    if rms > 0:
        return 20 * np.log10(rms + 1e-10)  # 使用更小的安全值
    else:
        return -80.0  # 返回一个合理的静音分贝值而不是负无穷


def calculate_audio_features(chunk):
    """计算 VAD 决策使用的帧级音频特征，不修改音频波形。"""
    abs_chunk = np.abs(chunk)
    return {
        "volume_db": calculate_db(chunk),
        "peak": float(abs_chunk.max()) if len(abs_chunk) else 0.0,
        "mean_abs": float(abs_chunk.mean()) if len(abs_chunk) else 0.0,
    }


class VADDecisionGate:
    """基于 Silero 概率、帧能量和自适应噪声底的 VAD 命中判定。"""

    def __init__(
        self,
        enter_prob_threshold: float,
        exit_prob_threshold: float,
        min_volume_db: float,
        snr_margin_db: float,
        active_snr_margin_db: float,
        noise_floor_window: int,
        noise_update_prob_threshold: float,
    ):
        self.enter_prob_threshold = enter_prob_threshold
        self.exit_prob_threshold = exit_prob_threshold
        self.min_volume_db = min_volume_db
        self.snr_margin_db = snr_margin_db
        self.active_snr_margin_db = active_snr_margin_db
        self.noise_update_prob_threshold = noise_update_prob_threshold
        self.noise_floor_db_values = deque(maxlen=noise_floor_window)

    def _noise_floor_db(self):
        if not self.noise_floor_db_values:
            return None
        return float(np.percentile(self.noise_floor_db_values, 20))

    def _energy_threshold(self, state):
        noise_floor_db = self._noise_floor_db()
        if noise_floor_db is None:
            return self.min_volume_db, None

        margin_db = self.snr_margin_db if state == State.IDLE else self.active_snr_margin_db
        return max(self.min_volume_db, noise_floor_db + margin_db), noise_floor_db

    def _update_noise_floor(self, volume_db):
        self.noise_floor_db_values.append(volume_db)

    def decide(self, state, smoothed_prob, features):
        volume_db = features["volume_db"]
        energy_threshold_db, noise_floor_db = self._energy_threshold(state)

        if state == State.IDLE:
            prob_threshold = self.enter_prob_threshold
            is_prob_hit = smoothed_prob > prob_threshold
            is_energy_hit = volume_db >= energy_threshold_db
            is_hit = is_prob_hit and is_energy_hit

            if not is_hit and smoothed_prob <= self.noise_update_prob_threshold:
                self._update_noise_floor(volume_db)
        else:
            prob_threshold = self.exit_prob_threshold
            is_prob_hit = smoothed_prob > prob_threshold
            is_energy_hit = volume_db >= energy_threshold_db
            is_hit = is_prob_hit and is_energy_hit

        return {
            "is_hit": is_hit,
            "prob_threshold": prob_threshold,
            "is_prob_hit": is_prob_hit,
            "energy_threshold_db": energy_threshold_db,
            "is_energy_hit": is_energy_hit,
            "noise_floor_db": noise_floor_db,
        }


class VADProcessor:
    """VAD处理器类，封装VAD计算过程"""

    # 类变量 - VAD配置常量
    SAMPLING_RATE = 16000
    DATA_TYPE = np.float32
    CHANNELS = 1  # 单声道
    WINDOW_SIZE = 512 if SAMPLING_RATE == 16000 else 256  # 32ms * 16000 / 1000

    def __init__(
        self,
        smoothing_window: int,
        prob_threshold: float,
        exit_prob_threshold: float,
        required_hits: int,
        required_misses: int,
        prebuffer: int,
        lang: str,
        volume_threshold: float,
        snr_margin_db: float,
        active_snr_margin_db: float,
        noise_floor_window: int,
        noise_update_prob_threshold: float,
        asr_url: str,
        vad_silence_threshold: float,
        enable_speaker_verification: bool = False,
        smoothing_method: str = "mean",
        enable_early_onset: bool = False,
        early_onset_floor: float = 0.35,
        early_onset_release: int = 2,
        early_onset_max_lookback: int = 16,
        enable_tail_extend: bool = False,
        tail_floor: float = 0.15,
        tail_release: int = 2,
        tail_max_lookahead: int = None,
        asr_hotwords: list = None,
    ):
        # 加载VAD模型
        self.prob_model = load_silero_vad(onnx=True)
        self.lang = lang
        self.asr_url = asr_url
        self.asr_hotwords = list(asr_hotwords) if asr_hotwords else []
        self.http_client = httpx.Client(timeout=30.0)

        # 加载初始状态
        self.state = State.IDLE

        # 音频缓冲区
        self.audio_buffer = np.array([], dtype=self.DATA_TYPE)

        # 移动平均窗口和阈值
        self.prob_window = deque(maxlen=smoothing_window)
        self.smoothing_method = smoothing_method
        self._ema_prob = None
        self.prob_threshold = prob_threshold

        # VAD 命中决策：进入语音段时门槛更高，语音段内门槛更宽松
        self.decision_gate = VADDecisionGate(
            enter_prob_threshold=prob_threshold,
            exit_prob_threshold=exit_prob_threshold,
            min_volume_db=volume_threshold,
            snr_margin_db=snr_margin_db,
            active_snr_margin_db=active_snr_margin_db,
            noise_floor_window=noise_floor_window,
            noise_update_prob_threshold=noise_update_prob_threshold,
        )

        # 兼容旧启动参数；新逻辑不再对 VAD 音频做逐采样点硬置零
        self.vad_silence_threshold = vad_silence_threshold

        # 状态转换参数
        self.required_hits = required_hits
        self.required_misses = required_misses

        # 预缓冲参数和循环缓冲区
        self.prebuffer = prebuffer
        self.prebuffer_queue = deque(maxlen=required_hits + prebuffer)

        # early-onset 辅助回溯：独立历史环形缓冲，存原始概率以便事后往回找真正起点。
        # 与 prebuffer_queue/状态机解耦；长度比基线起点多 max_lookback 帧，用于"额外往前"回溯。
        self.enable_early_onset = enable_early_onset
        self.early_onset_floor = early_onset_floor
        self.early_onset_release = early_onset_release
        self.early_onset_max_lookback = early_onset_max_lookback
        self.history_buffer = deque(
            maxlen=required_hits + prebuffer + early_onset_max_lookback
        )
        # watermark：上一段已送 ASR 的音频在 history 里的右端绝对序号，回溯不得越过它
        self.global_chunk_index = 0
        self.last_segment_end_index = -1

        # tail-extend 段尾延伸：ACTIVE-miss 的 chunk 存入此独立缓冲(含原始概率)，
        # 段结束时从最后命中往后做前向延伸，找回被切掉的尾音。最长 required_misses 个。
        self.enable_tail_extend = enable_tail_extend
        self.tail_floor = tail_floor
        self.tail_release = tail_release
        # 上限默认 = required_misses，且强制不超过它(tail_buffer 也只装这么多)
        self.tail_max_lookahead = min(
            tail_max_lookahead if tail_max_lookahead is not None else required_misses,
            required_misses,
        )
        self.tail_buffer = deque(maxlen=required_misses)

        # 状态转换计数器
        self.hit_count = 0
        self.miss_count = 0

        # VAD 决策诊断日志（每 N 次打印一次，便于调试自适应门限）
        self._vad_chunk_log_counter = 0
        self._vad_chunk_log_interval = 50

        # 语音段缓冲：收集完整语音段的音频数据
        self.speech_segment_buffer = []

        # 保存目录和计数器
        self.save_dir = "saved_audio_segments"
        os.makedirs(self.save_dir, exist_ok=True)
        self.segment_counter = 0

        # 声纹识别功能
        self.enable_speaker_verification = enable_speaker_verification
        if self.enable_speaker_verification:
            self.speaker_config = SystemConfig()
            self.embedding_manager = EmbeddingManager(self.speaker_config)
            logger.info("声纹识别功能已启用")

    def get_smooth_values(self, prob):
        self.prob_window.append(prob)
        if getattr(self, "smoothing_method", "mean") == "ema":
            win = max(1, self.prob_window.maxlen or 1)
            alpha = 2.0 / (win + 1.0)
            self._ema_prob = float(prob) if self._ema_prob is None else alpha * float(prob) + (1.0 - alpha) * self._ema_prob
            return self._ema_prob
        smoothed_prob = np.mean(self.prob_window)
        return smoothed_prob

    @staticmethod
    def encode_audio_to_base64_wav(audio_tensor: torch.Tensor, sample_rate: int) -> str:
        """将 torch tensor 编码为 base64 WAV 字符串"""
        buf = io.BytesIO()
        torchaudio.save(buf, audio_tensor.unsqueeze(0), sample_rate, format="wav")
        return base64.b64encode(buf.getvalue()).decode()

    def call_asr_api(self, merged_audio: torch.Tensor) -> dict:
        """调用远程 ASR REST API，返回响应 JSON"""
        bef_encode_ts = time.time()
        audio_b64 = self.encode_audio_to_base64_wav(merged_audio, self.SAMPLING_RATE)
        after_encode_ts = time.time()
        logger.info(f"音频编码耗时: {(after_encode_ts - bef_encode_ts) * 1000:.2f}ms (音频时长: {len(merged_audio) / self.SAMPLING_RATE:.2f}s, base64大小: {len(audio_b64) / 1024:.1f}KB)")
        lang = self.lang if self.lang != "auto" else None
        response = self.http_client.post(
            f"{self.asr_url}/asr/transcribe",
            json={
                "audio_base64": audio_b64,
                "format": "wav",
                "language": lang,
                "terms": self.asr_hotwords,
            },
        )
        response.raise_for_status()
        return response.json()

    def transcrib_and_save_speech_segment(self):
        bef_transcrib_ts = time.time()
        """转写并保存完整的语音段"""
        if not self.speech_segment_buffer:
            logger.warning("No speech segment to save")
            return None, None, None

        try:
            # 合并所有音频tensor
            audio_tensors = [chunk["audio_tensor"] for chunk in self.speech_segment_buffer]
            merged_audio = torch.cat(audio_tensors, dim=0)
            if self.enable_speaker_verification and self.embedding_manager:
                audio_numpy = merged_audio.numpy()
                hit, speaker_id, is_new_speaker = self.embedding_manager.verify_and_register(audio_numpy)
            else:
                speaker_id = ""

            bef_asr_ts = time.time()
            asr_response = self.call_asr_api(merged_audio)
            after_asr_ts = time.time()
            asr_result = asr_response["results"][0]

            start_timestamp = self.speech_segment_buffer[-1]["timestamp"]
            segment_datetime = datetime.fromtimestamp(start_timestamp)
            timestamp_str = segment_datetime.strftime("%Y%m%d_%H%M%S")

            self.segment_counter += 1
            filename = f"segment_{timestamp_str}_{self.segment_counter:03d}.wav"
            filepath = os.path.join(self.save_dir, filename)

            audio_to_save = merged_audio.unsqueeze(0)
            torchaudio.save(filepath, audio_to_save, self.SAMPLING_RATE)

            duration = len(merged_audio) / self.SAMPLING_RATE

            all_done_ts = time.time()
            received_ts = self.speech_segment_buffer[-1]["received_ts"]
            bef_process_ts = self.speech_segment_buffer[-1]["bef_process_ts"]
            logger.info(f"Saved speech segment: {filename} (duration: {duration:.3f}s)")
            logger.info(f"""ASR 处理时间: {after_asr_ts - bef_asr_ts:.3f}s
总时间: {all_done_ts - received_ts:.3f}s
receive时间: {received_ts - start_timestamp:.3f}s
receive到process前: {bef_process_ts - received_ts:.3f}s
process到transcribe前: {bef_transcrib_ts - bef_process_ts:.3f}s
transcribe到处理完毕: {all_done_ts - bef_transcrib_ts:.3f}s
""")
            return asr_result, start_timestamp, speaker_id
        except Exception as e:
            logger.opt(exception=True).error("Failed to save speech segment")
            return None, None, None
        finally:
            # 段已送 ASR：推进 watermark 到本段最后一个 chunk，回溯不得越过它取已用音频
            if self.enable_early_onset and self.speech_segment_buffer:
                self.last_segment_end_index = self.speech_segment_buffer[-1]['index']
            self.speech_segment_buffer.clear()

    def _collect_early_onset_prefix(self, baseline_start_index):
        """基线确认 ACTIVE 后，用原始概率从基线起点往左回溯，返回应前置到语音段的更早 chunk 列表。

        只扩不缩：回溯起点 <= baseline_start_index；遇连续 release 帧低于 floor、
        越过 max_lookback、或触及上一段 watermark 即停。返回按时间正序的 chunk 列表（可能为空）。
        """
        # history 里 index >= baseline_start_index 的部分由 prebuffer_queue 负责，这里只看更早的
        earlier = [c for c in self.history_buffer if c['index'] < baseline_start_index]
        if not earlier:
            return []

        # 从近到远扫描：用 release 容忍语音内部的短低谷，跨过它继续往前；
        # 连续 release 帧低于地板则认定到了段开头。onset_pos 记录"最后一个 >=floor 的位置"，
        # 即真正起点——prefix 一定截止到它，末尾不留任何低于地板帧。
        reversed_earlier = list(reversed(earlier))
        silent = 0
        onset_pos = -1  # earlier 倒序列表里的下标；-1 表示没找到任何语音帧
        for pos, chunk in enumerate(reversed_earlier):
            if pos >= self.early_onset_max_lookback:
                break
            if chunk['index'] <= self.last_segment_end_index:
                break  # 硬左界：不越过上一段已送 ASR 的音频
            if chunk['prob_raw'] >= self.early_onset_floor:
                onset_pos = pos
                silent = 0
            else:
                silent += 1
                if silent >= self.early_onset_release:
                    break

        if onset_pos < 0:
            return []
        # 取 [0, onset_pos] 这段（倒序），反转成时间正序前置到语音段
        prefix = reversed_earlier[:onset_pos + 1]
        prefix.reverse()
        return prefix

    def _collect_tail_extension(self):
        """段结束时，用原始概率从最后命中往后延伸，返回应追加到语音段尾的尾音 chunk 列表。

        镜像 _collect_early_onset_prefix：tail_buffer 按时间正序存了最后命中之后的 miss chunk。
        从头往后扫，prob_raw>=floor 记为尾音终点；连续 release 帧低于 floor、或越过
        max_lookahead 即停。返回 [开头, 尾音终点] 这段(时间正序，可能为空)，末尾静音不收。
        """
        if not self.tail_buffer:
            return []

        silent = 0
        end_pos = -1  # tail_buffer 里最后一个 >=floor 的位置；-1 表示无尾音，不延伸
        for pos, chunk in enumerate(self.tail_buffer):
            if pos >= self.tail_max_lookahead:
                break
            if chunk['prob_raw'] >= self.tail_floor:
                end_pos = pos
                silent = 0
            else:
                silent += 1
                if silent >= self.tail_release:
                    break

        if end_pos < 0:
            return []
        # 取 [0, end_pos] 这段(已是时间正序)，追加到语音段尾，末尾低于地板帧不收
        return list(self.tail_buffer)[:end_pos + 1]

    def process_audio_chunk(self, audio_data, timestamp, received_ts):
        """
        处理音频块，使用生成器逐个返回VAD结果

        Args:
            audio_data: 音频数据列表
            timestamp: 音频块的时间戳（必须提供）

        Yields:
            dict: VAD结果字典
        """
        bef_process_ts = time.time()
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
            # 提取一个完整的音频块（原始，用于 ASR）
            audio_chunk = self.audio_buffer[:self.WINDOW_SIZE]
            self.audio_buffer = self.audio_buffer[self.WINDOW_SIZE:]

            # VAD 决策路径只计算特征，不修改音频波形；ASR 仍使用原始 audio_chunk
            audio_features = calculate_audio_features(audio_chunk)
            volume_db = audio_features["volume_db"]

            # 诊断：chunk 幅度和自适应门限，用于调试 VAD 决策
            self._vad_chunk_log_counter += 1

            # ASR 用的原始 tensor，存入 speech_segment_buffer
            audio_tensor = torch.from_numpy(audio_chunk)

            # 使用 prob_model 计算语音概率（VAD 路径：不再做逐采样点硬置零）
            bef_vad_ts = time.time()
            speech_prob = self.prob_model(torch.from_numpy(audio_chunk), self.SAMPLING_RATE).item()
            after_vad_ts = time.time()
            smoothed_prob = self.get_smooth_values(speech_prob)

            # 状态管理逻辑：Silero 概率 + 自适应能量门控 + ACTIVE/IDLE 滞回
            vad_decision = self.decision_gate.decide(self.state, smoothed_prob, audio_features)
            is_hit = vad_decision["is_hit"]
            if self._vad_chunk_log_counter % self._vad_chunk_log_interval == 0:
                logger.info(
                    f"[vad gate] state={self.state.name} prob={speech_prob:.4f} smooth={smoothed_prob:.4f} "
                    f"prob_thr={vad_decision['prob_threshold']:.3f} volume={volume_db:.2f}dB "
                    f"energy_thr={vad_decision['energy_threshold_db']:.2f}dB "
                    f"noise_floor={vad_decision['noise_floor_db']} "
                    f"peak={audio_features['peak']:.4f} mean_abs={audio_features['mean_abs']:.4f} "
                    f"is_hit={is_hit}"
                )
            logger.trace(f"VAD 计算时间: {after_vad_ts - bef_vad_ts:.4f}s, 概率: {speech_prob:.4f}, 音量: {volume_db:.3f} dB")
            logger.trace(f'is_hit: {is_hit}')
            logger.trace(f'hit_count: {self.hit_count}')
            logger.trace(f'miss_count: {self.miss_count}')

            # 创建当前音频块的信息（包含torch tensor和时间戳）
            current_chunk = {
                "audio_tensor": audio_tensor,
                "timestamp": timestamp,  # 比较粗糙的时间戳
                "received_ts": received_ts,
                'bef_process_ts': bef_process_ts,
                'prob_raw': speech_prob,           # 未平滑概率，供 early-onset 回溯
                'index': self.global_chunk_index,  # 全局递增序号，做回溯左界判定
            }
            self.global_chunk_index += 1

            # 将当前音频块添加到预缓冲循环队列中（无论是否hit都要保存）
            self.prebuffer_queue.append(current_chunk)

            # early-onset：维护更长的历史环形缓冲（含原始概率），供基线确认后回溯精修起点
            if self.enable_early_onset:
                self.history_buffer.append(current_chunk)

            if is_hit:
                self.hit_count += 1
                self.miss_count = 0  # 重置miss计数器

                # 在IDLE状态下hit
                if self.state == State.IDLE:

                    # 检查是否达到转换条件
                    if self.hit_count >= self.required_hits:
                        self.state = State.ACTIVE

                        # 获取语音段开始时间戳（使用第一个命中的音频块时间戳）
                        # 注意：timestamp 暂不随 early-onset 前移，维持基线起点（见 TODO）
                        speech_start_timestamp = self.prebuffer_queue[-self.hit_count]['timestamp']

                        yield {
                            "type": "vad",
                            'timestamp': speech_start_timestamp
                            }

                        # early-onset：基线起点 = prebuffer_queue 最左 chunk 的全局序号，
                        # 在倒入 prebuffer 之前，先把回溯到的更早 chunk 前置进语音段
                        if self.enable_early_onset and self.prebuffer_queue:
                            baseline_start_index = self.prebuffer_queue[0]['index']
                            prefix = self._collect_early_onset_prefix(baseline_start_index)
                            if prefix:
                                self.speech_segment_buffer.extend(prefix)
                                logger.info(
                                    f"[early-onset] 起点前移 {len(prefix)} 帧 "
                                    f"({len(prefix) * VADProcessor.WINDOW_SIZE / VADProcessor.SAMPLING_RATE * 1000:.0f}ms)"
                                )

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

                    # tail-extend：重新命中→刚才那批 miss 是句内停顿(已随 prebuffer 进段)，
                    # 不是段尾，tail_buffer 作废重来
                    if self.enable_tail_extend:
                        self.tail_buffer.clear()

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

                    # tail-extend：ACTIVE-miss 的 chunk 存入尾部缓冲，供段结束时前向延伸找回尾音
                    if self.enable_tail_extend:
                        self.tail_buffer.append(current_chunk)

                    # 检查是否达到转换条件
                    if self.miss_count >= self.required_misses:
                        self.state = State.IDLE

                        # tail-extend：送 ASR 前，把尾部 miss 里仍属于尾音的 chunk 追加进语音段
                        if self.enable_tail_extend:
                            extension = self._collect_tail_extension()
                            if extension:
                                self.speech_segment_buffer.extend(extension)
                                logger.info(
                                    f"[tail-extend] 段尾延伸 {len(extension)} 帧 "
                                    f"({len(extension) * VADProcessor.WINDOW_SIZE / VADProcessor.SAMPLING_RATE * 1000:.0f}ms)"
                                )
                            self.tail_buffer.clear()

                        # 保存完整的语音段
                        asr_result, asr_timestamp, speaker_id = self.transcrib_and_save_speech_segment()
                        if asr_result is None:
                            logger.error('未能生成文本')
                            yield {
                                'type': 'error',
                                'error': '未能生成文本，请查看服务器端状况'
                                }
                        else:
                            yield {
                                'type': 'asr',
                                'asr_result': asr_result,
                                'timestamp': asr_timestamp,
                                'speaker_id': speaker_id
                                }
                            logger.info(f'生成文本：{asr_result["text"]}')


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


# 允许 session.init 按连接覆盖的参数白名单（= AppConfig 字段，排除部署级 asr_url）
_OVERRIDABLE_FIELDS = (
    'prob_threshold', 'exit_prob_threshold', 'smoothing_window', 'smoothing_method',
    'required_hits', 'required_misses', 'prebuffer', 'lang',
    'volume_threshold', 'snr_margin_db', 'active_snr_margin_db',
    'noise_floor_window', 'noise_update_prob_threshold', 'vad_silence_threshold',
    'enable_speaker_verification',
    'enable_early_onset', 'early_onset_floor', 'early_onset_release', 'early_onset_max_lookback',
    'enable_tail_extend', 'tail_floor', 'tail_release', 'tail_max_lookahead',
)


def _base_params_from_config():
    """全局默认参数快照，作为每连接配置的基线。"""
    return {f: getattr(config, f) for f in _OVERRIDABLE_FIELDS}


def _validate_params(p):
    """逐条复用启动期约束校验一份完整参数 dict；不满足抛 ValueError(中文原因)。"""
    if p['prebuffer'] + p['required_hits'] > p['required_misses']:
        raise ValueError("prebuffer + required_hits 必须 <= required_misses")
    if not p['noise_floor_window'] > 0:
        raise ValueError("noise_floor_window 必须 > 0")
    if p['exit_prob_threshold'] > p['prob_threshold']:
        raise ValueError("exit_prob_threshold 必须 <= prob_threshold")
    if p['active_snr_margin_db'] > p['snr_margin_db']:
        raise ValueError("active_snr_margin_db 必须 <= snr_margin_db")
    if p['early_onset_release'] < 1:
        raise ValueError("early_onset_release 必须 >= 1")
    if p['early_onset_max_lookback'] < 0:
        raise ValueError("early_onset_max_lookback 必须 >= 0")
    if p['tail_release'] < 1:
        raise ValueError("tail_release 必须 >= 1")
    if not (p['tail_max_lookahead'] is None or p['tail_max_lookahead'] >= 0):
        raise ValueError("tail_max_lookahead 必须 >= 0 或省略")


def _merge_init_config(overrides):
    """把 session.init 的 config 覆盖到全局默认上，返回完整参数 dict。
    未知字段 → 日志告知后忽略(允许上游透传非 VAD 字段，如 asr_hotwords);
    类型不匹配 / 约束不过 → 抛 ValueError(供调用方回错关连接)。"""
    if not isinstance(overrides, dict):
        raise ValueError("config 必须是对象")
    unknown = set(overrides) - set(_OVERRIDABLE_FIELDS)
    if unknown:
        logger.info(f"session.init 含非 VAD 字段，已忽略: {sorted(unknown)}")

    base = _base_params_from_config()
    for k, v in overrides.items():
        if k not in _OVERRIDABLE_FIELDS:
            continue
        if k == 'tail_max_lookahead' and v is None:
            base[k] = None
            continue
        ref = base[k]
        # 按基线值的类型做轻校验：bool/数值/字符串各自匹配，类型不符直接拒绝
        if isinstance(ref, bool):
            if not isinstance(v, bool):
                raise ValueError(f"{k} 应为布尔值")
        elif isinstance(ref, str):
            if not isinstance(v, str):
                raise ValueError(f"{k} 应为字符串")
        elif isinstance(ref, (int, float)) or ref is None:
            # 数值字段(部分默认为 None,如 tail_max_lookahead)：接受 int/float，拒绝 bool
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{k} 应为数值")
        base[k] = v

    _validate_params(base)
    return base


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted - NEW VERSION")

    # 等首帧最多 1 秒以区分新/老客户端：
    #   session.init → 用其 config 覆盖默认；audio_chunk → 用默认并补处理这帧；
    #   超时(老客户端在等 ready) → 用默认。
    try:
        first = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
    except asyncio.TimeoutError:
        first = None
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed before first message")
        return

    pending_audio = None  # 首帧若是 audio_chunk，建好 processor 后再处理
    if first is not None and first.get("type") == "session.init":
        init_config = first.get("config") or {}
        logger.info(f"收到 session.init 配置: {init_config}")
        # asr_hotwords 是会话级、无全局默认,不走 VAD 白名单
        asr_hotwords = init_config.pop("asr_hotwords", None)
        try:
            params = _merge_init_config(init_config)
        except ValueError as e:
            logger.info(f"session.init 配置无效，关闭连接: {e}")
            await websocket.send_json({"type": "error", "error": f"session.init 配置无效: {e}"})
            await websocket.close()
            return
        logger.info("session.init 已应用按连接配置")
    else:
        params = _base_params_from_config()
        asr_hotwords = None
        if first is not None and first.get("type") == "audio_chunk":
            pending_audio = first

    # 为该连接创建独立 VAD 处理器（参数 = 默认或 session.init 覆盖后）
    logger.info("Creating VAD processor for new connection...")
    vad_processor = VADProcessor(asr_url=config.asr_url, asr_hotwords=asr_hotwords, **params)
    logger.info("WebSocket connection established with dedicated VAD processor")

    # 就绪状态：带回本连接实际生效的全部参数，供客户端核对
    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "message": "VAD 和 ASR 模型加载完成，可以开始发送音频数据",
        "effective_config": {**params, "asr_hotwords": vad_processor.asr_hotwords},
        })
    logger.info("Ready status sent")

    async def handle_audio(message, received_ts):
        audio_data = message.get("data")
        timestamp = message.get("timestamp")
        if audio_data is None:
            return
        for result in vad_processor.process_audio_chunk(audio_data, timestamp, received_ts):
            await websocket.send_json(result)

    try:
        # 先补处理首帧 audio_chunk（若有）
        if pending_audio is not None:
            await handle_audio(pending_audio, time.time())

        while True:
            message = await websocket.receive_json()
            received_ts = time.time()
            logger.trace(f"Received")

            if message.get("type") == "audio_chunk":
                await handle_audio(message, received_ts)
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
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'TRACE'],
                        help='日志文件日志级别（STDOUT不受影响）')
    parser.add_argument('--uvicorn-log-level', default='info')
    parser.add_argument('--prob-threshold', type=float, default=0.4,  # 线上默认
                        help='IDLE 状态进入 ACTIVE 的移动平均概率阈值')
    parser.add_argument('--exit-prob-threshold', type=float, default=0.25,
                        help='ACTIVE 状态维持语音段的移动平均概率阈值，低于进入阈值以保留尾音')
    parser.add_argument('--smoothing-window', type=int, default=6,  # 线上默认
                        help='平滑窗口(线上默认win=6)')
    parser.add_argument('--smoothing-method', type=str, default='ema', choices=['mean', 'ema'],
                        help='概率平滑:mean=滑窗均值(线上原版)/ema=指数平滑(本服务新增,降误触发)')
    parser.add_argument('--required-hits', type=int, default=3,  # 旧值: 5
                        help='从IDLE进入ACTIVE状态需要的连续命中次数')
    parser.add_argument('--required-misses', type=int, default=20,  # 评测那套(线上32)
                        help='从ACTIVE进入IDLE状态需要的连续未命中次数')
    parser.add_argument('--prebuffer', type=int, default=16,  # 旧值: 12
                        help='从IDLE转换到ACTIVE时包含的之前音频块数量')
    parser.add_argument('--lang', type=str, default='auto',
                        help='ASR语言')
    parser.add_argument('--volume-threshold', type=float, default=-30.0,  # 评测那套(线上-38/-50)
                        help='最低音量门限值（分贝），自适应噪声门限不会低于此值')
    parser.add_argument('--snr-margin-db', type=float, default=28.0,  # 评测那套(线上10)
                        help='IDLE 状态下，进入语音段需要高于自适应噪声底的分贝数')
    parser.add_argument('--active-snr-margin-db', type=float, default=0.0,
                        help='ACTIVE 状态下，维持语音段需要高于自适应噪声底的分贝数，可低于进入门限以保留尾音')
    parser.add_argument('--noise-floor-window', type=int, default=120,  # 评测那套(线上80)
                        help='估计自适应噪声底使用的最近静音 chunk 数')
    parser.add_argument('--noise-update-prob-threshold', type=float, default=0.5,  # 评测那套(线上0.2)
                        help='仅当平滑 VAD 概率低于该值时，才用当前 chunk 更新噪声底')
    parser.add_argument('--vad-silence-threshold', type=float, default=0.05,
                        help='兼容旧参数；当前版本不再用它做逐采样点硬置零')
    parser.add_argument('--enable_speaker_verification', action='store_true',
                        help='启用声纹识别功能')
    # early-onset 辅助回溯：用原始(未 EMA)概率把语音段左边界往前精修，捞回被切掉的前导字
    parser.add_argument('--enable-early-onset', action='store_true',
                        help='启用 early-onset 起点回溯精修（默认关，关闭时行为与现状一致）')
    parser.add_argument('--early-onset-floor', type=float, default=0.01,  # 线上默认
                        help='回溯时判定"还在语音里"的原始概率地板（低于基线 prob-threshold）')
    parser.add_argument('--early-onset-release', type=int, default=10,  # 线上默认
                        help='连续多少帧原始概率低于地板，才认定退到段开头并停止回溯')
    parser.add_argument('--early-onset-max-lookback', type=int, default=20,  # 线上默认
                        help='在基线起点(prebuffer 覆盖)基础上，额外往前回溯的最大帧数')
    # tail-extend 段尾延伸：用原始概率把语音段右边界往后延伸，找回被切掉的尾音(镜像 early-onset)
    parser.add_argument('--enable-tail-extend', action='store_true',
                        help='启用 tail-extend 段尾延伸（默认关，关闭时行为与现状一致）')
    parser.add_argument('--tail-floor', type=float, default=0.01,  # 线上默认
                        help='段尾延伸判定"还是尾音"的原始概率地板。需 < exit-prob-threshold(默认0.25)'
                             '才有发挥空间：>exit 的尾音基线在 ACTIVE 已自己收进段，进 tail_buffer 的都 <exit')
    parser.add_argument('--tail-release', type=int, default=10,  # 线上默认
                        help='连续多少帧原始概率低于地板，才认定到段尾并停止延伸')
    parser.add_argument('--tail-max-lookahead', type=int, default=20,  # 线上默认
                        help='段尾最多延伸的帧数，默认=required_misses，且不超过它')
    parser.add_argument('--asr-url', type=str, required=True,
                        help='ASR REST API 服务地址，例如 http://localhost:50300')

    args = parser.parse_args()

    assert args.prebuffer + args.required_hits <= args.required_misses
    assert args.noise_floor_window > 0
    assert args.exit_prob_threshold <= args.prob_threshold
    assert args.active_snr_margin_db <= args.snr_margin_db
    assert args.early_onset_release >= 1
    assert args.early_onset_max_lookback >= 0
    assert args.tail_release >= 1
    assert args.tail_max_lookahead is None or args.tail_max_lookahead >= 0

    # 配置 loguru 日志 - 最简单配置：控制台 + 文件
    logger.add("vad_server.log", level=args.log_level)

    # 设置全局配置
    config.prob_threshold = args.prob_threshold
    config.exit_prob_threshold = args.exit_prob_threshold
    config.smoothing_window = args.smoothing_window
    config.smoothing_method = args.smoothing_method
    config.required_hits = args.required_hits
    config.required_misses = args.required_misses
    config.prebuffer = args.prebuffer
    config.lang = args.lang
    config.volume_threshold = args.volume_threshold
    config.snr_margin_db = args.snr_margin_db
    config.active_snr_margin_db = args.active_snr_margin_db
    config.noise_floor_window = args.noise_floor_window
    config.noise_update_prob_threshold = args.noise_update_prob_threshold
    config.enable_speaker_verification = args.enable_speaker_verification
    config.asr_url = args.asr_url.rstrip("/")
    config.vad_silence_threshold = args.vad_silence_threshold
    config.enable_early_onset = args.enable_early_onset
    config.early_onset_floor = args.early_onset_floor
    config.early_onset_release = args.early_onset_release
    config.early_onset_max_lookback = args.early_onset_max_lookback
    config.enable_tail_extend = args.enable_tail_extend
    config.tail_floor = args.tail_floor
    config.tail_release = args.tail_release
    config.tail_max_lookahead = args.tail_max_lookahead

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
        log_level=args.uvicorn_log_level
        )
