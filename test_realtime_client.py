"""
实时麦克风 VAD+ASR 测试客户端

使用麦克风采集音频，通过 WebSocket 发送到 VAD 服务器，实时显示识别结果。
按 Ctrl+C 退出。

用法:
    uv run test_realtime_client.py
    uv run test_realtime_client.py --uri ws://remote-host:8000/ws
"""

import argparse
import asyncio
import json
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import websockets

SAMPLING_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
# sounddevice 回调的 blocksize，每次回调采集的采样点数
BLOCK_SIZE = 320


class RealtimeVADClient:

    def __init__(self, uri: str, init_config: dict = None):
        self.uri = uri
        # 每连接配置：发 session.init 携带；None 时发空 {} 用服务端全局默认（顺带省掉服务端 1s 兜底等待）
        self.init_config = init_config or {}
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.running = True

    async def run(self):
        print(f"连接到 {self.uri} ...")
        async with websockets.connect(self.uri) as ws:
            # 首帧发 session.init（可选覆盖本连接 VAD 参数），再等 ready
            await ws.send(json.dumps({"type": "session.init", "config": self.init_config}))
            ready_msg = json.loads(await ws.recv())
            if ready_msg.get("status") != "ready":
                raise RuntimeError(f"服务器未就绪/初始化失败: {ready_msg}")
            print(f"服务器就绪: {ready_msg.get('message')}")
            print(f"生效配置: {json.dumps(ready_msg.get('effective_config', {}), ensure_ascii=False)}")
            print("开始录音，请说话... (Ctrl+C 退出)\n")

            # 启动麦克风采集（在音频回调线程中写入 queue）
            loop = asyncio.get_running_loop()
            stream = sd.InputStream(
                samplerate=SAMPLING_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=lambda indata, frames, time_info, status: loop.call_soon_threadsafe(
                    self.audio_queue.put_nowait, (indata.copy(), time.time())
                ),
            )
            stream.start()

            try:
                sender = asyncio.create_task(self._send_loop(ws))
                receiver = asyncio.create_task(self._receive_loop(ws))
                await asyncio.gather(sender, receiver)
            finally:
                stream.stop()
                stream.close()

    async def _send_loop(self, ws):
        while self.running:
            try:
                indata, ts = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            # indata shape: (BLOCK_SIZE, 1) -> flatten
            chunk = indata.flatten()
            message = {
                "type": "audio_chunk",
                "data": chunk.tolist(),
                "timestamp": ts,
            }
            await ws.send(json.dumps(message))

    async def _receive_loop(self, ws):
        while self.running:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("连接已关闭")
                self.running = False
                return

            result = json.loads(raw)
            msg_type = result.get("type")

            if msg_type == "vad":
                # 静默处理，不打印每个 VAD 帧
                pass
            elif msg_type == "asr":
                now = datetime.now().strftime("%H:%M:%S")
                asr_result = result.get("asr_result", {})
                speaker = result.get("speaker_id", "")
                prefix = f"[{speaker}] " if speaker else ""
                emotion = asr_result.get("emotion", "")
                event = asr_result.get("event", "")
                lang = asr_result.get("language", "")
                tags = f"({lang}|{emotion}|{event})" if any([emotion, event, lang]) else ""
                print(f"[{now}] {prefix}{asr_result.get('text', '')} {tags}")
            elif msg_type == "error":
                print(f"[错误] {result.get('error')}")
            elif msg_type == "status":
                print(f"[状态] {result.get('message')}")

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="实时麦克风 VAD+ASR 测试客户端")
    parser.add_argument("--uri", default="ws://localhost:50160/ws", help="VAD WebSocket 服务地址")
    parser.add_argument("--init-config", default=None,
                        help='session.init 的 config(JSON)，覆盖本连接 VAD 参数，'
                             '如 \'{"prob_threshold":0.6,"enable_early_onset":true}\'；不传则用服务端默认')
    args = parser.parse_args()

    init_config = json.loads(args.init_config) if args.init_config else None
    client = RealtimeVADClient(args.uri, init_config=init_config)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
