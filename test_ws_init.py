"""
session.init 握手协议测试客户端（不发真音频，只验证连接初始化四条路径）

验证点：
  A. 发合法 session.init → 收到 ready，且 effective_config 反映覆盖值
  B. 发非法 session.init（exit>prob）→ 收到 error 且连接被关闭
  C. 不发 init、直接发一帧 audio_chunk → 收到 ready(默认配置)，连接保持
  D. 老客户端：连上什么都不发 → 服务端 ~1s 超时兜底后主动发 ready(默认配置)

用法:
    uv run test_ws_init.py --uri ws://localhost:50160/ws
"""

import argparse
import asyncio
import json
import time

import websockets


async def _recv(ws, timeout):
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


async def case_a(uri):
    """A: 合法 init → ready + effective_config 生效"""
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "config": {"prob_threshold": 0.6, "enable_early_onset": True, "smoothing_window": 4},
        }))
        msg = await _recv(ws, 30)  # 首次连接要加载模型，给足时间
        assert msg.get("status") == "ready", msg
        ec = msg.get("effective_config") or {}
        assert ec.get("prob_threshold") == 0.6, ec
        assert ec.get("enable_early_onset") is True, ec
        assert ec.get("smoothing_window") == 4, ec
        # 未覆盖项应为默认（exit_prob_threshold 线上默认 0.25）
        assert ec.get("exit_prob_threshold") == 0.25, ec
        print("✅ A 合法 init：ready 带 effective_config，覆盖生效、其余取默认")


async def case_b(uri):
    """B: 非法 init（exit>prob）→ error + 连接关闭"""
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "config": {"exit_prob_threshold": 0.9},  # > prob_threshold(0.4)
        }))
        msg = await _recv(ws, 30)
        assert msg.get("type") == "error", msg
        assert "session.init 配置无效" in msg.get("error", ""), msg
        # 连接应被服务端关闭
        try:
            await _recv(ws, 5)
            closed = False
        except websockets.exceptions.ConnectionClosed:
            closed = True
        assert closed, "非法 init 后连接应被关闭"
        print(f"✅ B 非法 init：收到 error 并关闭连接 -> {msg.get('error')}")


async def case_c(uri):
    """C: 不发 init，首帧直接 audio_chunk → ready(默认)，连接保持"""
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "data": [0.0] * 320,
            "timestamp": time.time(),
        }))
        msg = await _recv(ws, 30)
        assert msg.get("status") == "ready", msg
        ec = msg.get("effective_config") or {}
        assert ec.get("prob_threshold") == 0.4, ec  # 默认
        print("✅ C 首帧 audio_chunk：ready(默认配置)，连接正常")


async def case_d(uri):
    """D: 老客户端不发任何消息 → 服务端超时兜底后发 ready(默认)"""
    async with websockets.connect(uri) as ws:
        t0 = time.time()
        msg = await _recv(ws, 30)  # 含 1s 超时兜底 + 模型加载
        dt = time.time() - t0
        assert msg.get("status") == "ready", msg
        ec = msg.get("effective_config") or {}
        assert ec.get("prob_threshold") == 0.4, ec
        print(f"✅ D 老客户端兜底：{dt:.2f}s 后收到 ready(默认配置)，未死锁")


async def main():
    parser = argparse.ArgumentParser(description="session.init 握手协议测试")
    parser.add_argument("--uri", default="ws://localhost:50160/ws")
    args = parser.parse_args()

    for name, fn in [("A", case_a), ("B", case_b), ("C", case_c), ("D", case_d)]:
        try:
            await fn(args.uri)
        except Exception as e:
            print(f"❌ {name} 失败: {e!r}")
            raise
    print("\n全部握手用例通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())
