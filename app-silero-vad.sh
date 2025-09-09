#!/bin/bash

# SenceVoice启动脚本
# 用法: bash app-sencevoice.sh {start|stop|restart|debug} [cuda_device]

# ============ 配置区域 ============
SERVICE_NAME="Silero-vad"           # 服务名称
RUNTIME_DIR=".silero-vad"           # 运行时文件目录
PYTHON_CMD="uv run"                 # Python执行命令 (可改为: uv run python, python3, conda run -n myenv python 等)
SCRIPT_NAME="vad_websocket_server.py"         # 主脚本文件名
DEFAULT_CUDA_DEVICE=4               # 默认CUDA设备号
PORT=50160                          # 服务端口
PROB_THRESHOLD=0.4                  # 语音检测概率阈值
REQUIRED_HITS=3                     # 语音检测所需的连续高概率次数
REQUIRED_MISSES=16                    # 语音检测所需的连续低概率次数
PREBUFFER=12                        # 缓冲队列长度
VOLUME_THRESHOLD=-40                # 音量阈值
ENABLE_SPEAKER_VERIFICATION="--enable_speaker_verification"   # 启用说话人验证选项 (可选)
# ================================

SERVICE_NAME_LOWER=$(echo "$SERVICE_NAME" | tr '[:upper:]' '[:lower:]')
PID_FILE="$RUNTIME_DIR/${SERVICE_NAME_LOWER}.pid"
LOG_FILE="$RUNTIME_DIR/${SERVICE_NAME_LOWER}.log"

# 获取CUDA设备号
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

# 检查进程是否运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            return 0
        else
            rm -f $PID_FILE
            return 1
        fi
    fi
    return 1
}

# 创建运行时目录
create_runtime_dir() {
    if [ ! -d "$RUNTIME_DIR" ]; then
        mkdir -p "$RUNTIME_DIR"
        echo "创建运行时目录: $RUNTIME_DIR"
    fi
}

# 启动服务
start() {
    create_runtime_dir

    if is_running; then
        echo "$SERVICE_NAME 已经在运行中 (PID: $(cat $PID_FILE))"
        return 1
    fi

    echo "启动 $SERVICE_NAME (CUDA_VISIBLE_DEVICES=$CUDA_DEVICE)..."
    echo "使用命令: $PYTHON_CMD"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup $PYTHON_CMD $SCRIPT_NAME \
        --port $PORT \
        --prob-threshold $PROB_THRESHOLD \
        --required-hits $REQUIRED_HITS \
        --required-misses $REQUIRED_MISSES \
        --prebuffer $PREBUFFER \
        --volume-threshold $VOLUME_THRESHOLD \
        $ENABLE_SPEAKER_VERIFICATION \
        > $LOG_FILE 2>&1 & echo $! > $PID_FILE

    sleep 2
    if is_running; then
        echo "$SERVICE_NAME 启动成功 (PID: $(cat $PID_FILE))"
        echo "查看实时日志: tail -f $LOG_FILE"
    else
        echo "$SERVICE_NAME 启动失败，请检查日志: $LOG_FILE"
        return 1
    fi
}

# 停止服务
stop() {
    if ! is_running; then
        echo "$SERVICE_NAME 未在运行"
        return 0
    fi

    PID=$(cat $PID_FILE)
    echo "停止 $SERVICE_NAME (PID: $PID)..."
    kill $PID

    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            rm -f $PID_FILE
            echo "$SERVICE_NAME 已停止"
            return 0
        fi
        sleep 1
    done

    # 强制终止
    echo "强制终止 $SERVICE_NAME..."
    kill -9 $PID 2>/dev/null
    rm -f $PID_FILE
    echo "$SERVICE_NAME 已强制停止"
}

# 重启服务
restart() {
    echo "重启 $SERVICE_NAME..."
    stop
    sleep 2
    start
}

# 调试模式
debug() {
    if is_running; then
        echo "警告: $SERVICE_NAME 已在后台运行 (PID: $(cat $PID_FILE))"
        echo "请先执行 stop 命令停止后台服务"
        return 1
    fi

    echo "调试模式启动 $SERVICE_NAME (CUDA_VISIBLE_DEVICES=$CUDA_DEVICE)..."
    echo "使用命令: $PYTHON_CMD"
    echo "按 Ctrl+C 退出"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $PYTHON_CMD $SCRIPT_NAME \
        --port $PORT \
        --prob-threshold $PROB_THRESHOLD \
        --required-hits $REQUIRED_HITS \
        --required-misses $REQUIRED_MISSES \
        --prebuffer $PREBUFFER \
        --volume-threshold $VOLUME_THRESHOLD \
        $ENABLE_SPEAKER_VERIFICATION
}

# 显示状态
status() {
    if is_running; then
        PID=$(cat $PID_FILE)
        echo "$SERVICE_NAME 正在运行 (PID: $PID)"
        echo "日志文件: $LOG_FILE"
        echo "查看实时日志: tail -f $LOG_FILE"
    else
        echo "$SERVICE_NAME 未在运行"
    fi
}

# 显示使用方法
usage() {
    echo "用法: $0 {start|stop|restart|debug|status} [cuda_device]"
    echo ""
    echo "命令:"
    echo "  start   - 后台启动服务"
    echo "  stop    - 停止服务"
    echo "  restart - 重启服务"
    echo "  debug   - 调试模式(前台运行)"
    echo "  status  - 显示运行状态"
    echo ""
    echo "参数:"
    echo "  cuda_device - CUDA设备号 (默认: $DEFAULT_CUDA_DEVICE)"
    echo ""
    echo "示例:"
    echo "  $0 start      # 使用默认CUDA设备启动"
    echo "  $0 start 0    # 使用CUDA设备0启动"
    echo "  $0 debug 1    # 使用CUDA设备1调试模式启动"
}

# 主程序
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    debug)
        debug
        ;;
    status)
        status
        ;;
    *)
        usage
        exit 1
        ;;
esac

exit $?