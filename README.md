## 启动服务

```bash
# 1. 启动 ASR 服务（或使用其他兼容 POST /asr/transcribe 接口的服务）
uv run asr_server.py --port 10001

# 2. 启动 VAD WebSocket 服务，指向 ASR 服务
#uv run vad_websocket_server.py --asr-url http://localhost:10001 --port 50160
uv run vad_websocket_server.py --asr-url http://localhost:50300 --port 50160 --volume-threshold -38 --prob-threshold 0.5 --exit-prob-threshold 0.25 --snr-margin-db 28 --active-snr-margin-db 0 --noise-floor-window 120 --noise-update-prob-threshold 0.5

# 3. 实时麦克风测试
uv run test_realtime_client.py

# 或离线 WAV 文件测试
uv run test_client.py /path/to/wav/dir
```

## 安装依赖

```bash
uv sync
```
