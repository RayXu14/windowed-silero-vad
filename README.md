## 文件说明

### vad_websocket_server.py
VAD WebSocket 服务器，提供实时语音检测服务。

**使用方法：**
```bash
# 启动服务器
uv run vad_websocket_server.py

# 自定义参数启动示例（更多参数详见文件内命令行参数设定）
uv run vad_websocket_server.py --port 8000 --prob-threshold 0.8
```

### test_client.py
VAD 客户端测试程序，连接服务器并发送若干个wav文件拼合而成的音频数据进行检测。

**使用方法：**
```bash
# 运行客户端（需要先启动服务器）
uv run test_client.py
```

### analyze_wav.ipynb
用于分析 WAV 文件（分贝分布）和测试 VAD 模型效果（概率分布、延迟分布）

## 安装依赖

```bash
uv sync
```
