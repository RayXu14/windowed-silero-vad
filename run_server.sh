#!/bin/bash
# 启动 windowed-silero-vad-ema —— 线上 50160 的兼容增强版(加EMA概率平滑)。
# WebSocket协议/输入输出与线上完全一致,其他程序无需改动即可接入。
# 用线上 venv(环境一致,不重建)。
#
# 用法:
#   bash run_server.sh [port]              # 默认 G2: win=8 + ema(评测最优)
#   SMOOTH_WIN=6 bash run_server.sh        # G1: win=6 + ema
#   SMOOTH_METHOD=mean bash run_server.sh  # 退回线上原版平滑(滑窗均值)对照
#
# 参数都可命令行覆盖(和线上一样),下面只是常用默认。

cd "$(dirname "$0")"
VENV=/home/projects/webrtc-digital-human/standalone_AI_module_servers/windowed-silero-vad/.venv/bin/python
PORT="${1:-50161}"                          # 默认50161(不占线上50160)
SMOOTH_WIN="${SMOOTH_WIN:-8}"               # 默认 win=8 (G2)
SMOOTH_METHOD="${SMOOTH_METHOD:-ema}"       # 默认 ema
ASR_URL="${ASR_URL:-http://localhost:50300}"

export CUDA_VISIBLE_DEVICES=3
echo "启动 windowed-silero-vad-ema: ws://0.0.0.0:${PORT}/ws"
echo "  平滑: smoothing-window=${SMOOTH_WIN} smoothing-method=${SMOOTH_METHOD}"
echo "  ASR: ${ASR_URL} | 协议与线上50160完全兼容"
exec $VENV vad_websocket_server.py \
  --host 0.0.0.0 --port "${PORT}" \
  --asr-url "${ASR_URL}" \
  --smoothing-window "${SMOOTH_WIN}" \
  --smoothing-method "${SMOOTH_METHOD}"
