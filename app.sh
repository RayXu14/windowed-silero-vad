#!/bin/bash

# Silero-VAD启动脚本
# 用法: bash app.sh {start|stop|restart|debug|kill|status|logs} [cuda_device]

# ============ 配置区域 ============
SERVICE_NAME="Silero-vad"           # 服务名称
RUNTIME_DIR="logs"                  # 运行时文件目录
PYTHON_CMD="uv run"                 # Python执行命令 (可改为: uv run python, python3, conda run -n myenv python 等)
SCRIPT_NAME="vad_websocket_server.py"         # 主脚本文件名
DEFAULT_CUDA_DEVICE=5               # 默认CUDA设备号
PORT=50160                          # 服务端口
PROB_THRESHOLD=0.4                  # 语音检测概率阈值
REQUIRED_HITS=3                     # 语音检测所需的连续高概率次数
REQUIRED_MISSES=16                  # 语音检测所需的连续低概率次数
PREBUFFER=12                        # 缓冲队列长度
VOLUME_THRESHOLD=-50                # 音量阈值
# ENABLE_SPEAKER_VERIFICATION="--enable_speaker_verification"   # 启用说话人验证选项 (可选)
ENABLE_SPEAKER_VERIFICATION=""      # 启用说话人验证选项 (可选)
# ================================

PID_FILE="$RUNTIME_DIR/app.pid"
INFO_FILE="$RUNTIME_DIR/app.info"
LOG_FILE="$RUNTIME_DIR/app.log"

# 获取CUDA设备号
CUDA_DEVICE=${2:-$DEFAULT_CUDA_DEVICE}

# 检查进程是否运行（不删除文件）
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ps -p $PID > /dev/null 2>&1; then
            return 0
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

# 构建完整的启动命令
get_service_command() {
    cat << EOF | tr '\n' ' ' | tr -s ' '
$PYTHON_CMD $SCRIPT_NAME
--port $PORT
--prob-threshold $PROB_THRESHOLD
--required-hits $REQUIRED_HITS
--required-misses $REQUIRED_MISSES
--prebuffer $PREBUFFER
--volume-threshold $VOLUME_THRESHOLD
$ENABLE_SPEAKER_VERIFICATION
EOF
}

# 启动服务
start() {
    create_runtime_dir

    if is_running; then
        echo "$SERVICE_NAME 已经在运行中 (PID: $(cat $PID_FILE))"
        return 1
    fi

    # 先检查端口占用
    if command -v lsof > /dev/null 2>&1; then
        if lsof -i:$PORT > /dev/null 2>&1; then
            echo "错误: 端口 $PORT 已被占用"
            echo "请先使用 kill 命令清理残留进程"
            echo "查看占用端口的进程: lsof -i:$PORT"
            return 1
        fi
    fi

    # 端口检查通过后，删除旧的PID和INFO文件
    if [ -f "$PID_FILE" ] || [ -f "$INFO_FILE" ]; then
        echo "检测到旧的PID/INFO文件，正在清理..."
        rm -f "$PID_FILE" "$INFO_FILE"
    fi

    echo "启动 $SERVICE_NAME (CUDA_VISIBLE_DEVICES=$CUDA_DEVICE)..."
    echo "使用命令: $PYTHON_CMD"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup $(get_service_command) > $LOG_FILE 2>&1 & echo $! > $PID_FILE

    local pid=$(cat $PID_FILE)

    # 保存完整启动命令和启动时间到INFO文件
    cat > "$INFO_FILE" << EOF
PID=$pid
COMMAND=$(get_service_command)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE=$LOG_FILE
CUDA_DEVICE=$CUDA_DEVICE
EOF

    sleep 2
    if is_running; then
        echo "$SERVICE_NAME 启动成功 (PID: $(cat $PID_FILE))"
        echo "查看实时日志: tail -f $LOG_FILE"
    else
        echo "$SERVICE_NAME 启动失败，请检查日志: $LOG_FILE"
        rm -f "$PID_FILE" "$INFO_FILE"
        return 1
    fi
}

# 停止服务
stop() {
    # 检查PID文件是否存在
    if [ ! -f "$PID_FILE" ]; then
        echo "错误: PID文件不存在"
        echo "请使用 kill 命令处理可能的残留进程"
        return 1
    fi

    PID=$(cat $PID_FILE)

    # 检查PID对应的进程是否存在
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "错误: PID文件中的进程(PID: $PID)已不存在"
        echo "可能已被手动kill，请使用 kill 命令清理残留进程"
        return 1
    fi

    echo "停止 $SERVICE_NAME (PID: $PID)..."
    kill $PID

    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "$SERVICE_NAME 已停止"
            return 0
        fi
        sleep 1
    done

    # 强制终止
    echo "进程未响应TERM信号，强制终止..."
    kill -9 $PID 2>/dev/null
    sleep 1

    # stop成功也不删除文件
    echo "$SERVICE_NAME 已停止"
    return 0
}

# 强制终止服务（双重保险：PID+完整命令匹配）
kill_service() {
    local killed=false

    # 方案1：通过PID强制终止（先TERM再KILL）
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "通过PID终止进程 (PID: $pid)..."

            # 先发送SIGTERM，给进程清理机会
            kill -TERM "$pid" 2>/dev/null

            # 等待3秒
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 3 ]; do
                sleep 1
                count=$((count + 1))
            done

            # 如果还在运行，发送SIGKILL
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "进程未响应TERM信号，强制终止..."
                kill -KILL "$pid" 2>/dev/null
            fi

            killed=true
        fi
    fi

    # 方案2：通过完整命令匹配（备用方案）
    if [ -f "$INFO_FILE" ]; then
        local saved_command=$(grep "^COMMAND=" "$INFO_FILE" | cut -d'=' -f2-)
        if [ -n "$saved_command" ]; then
            echo "使用完整命令匹配查找进程..."
            local pids=$(pgrep -f "$saved_command")

            if [ -n "$pids" ]; then
                echo "找到匹配的进程: $pids"
                # 同样先TERM再KILL
                echo "$pids" | xargs kill -TERM 2>/dev/null
                sleep 2
                # 检查是否还在运行
                for p in $pids; do
                    if ps -p "$p" > /dev/null 2>&1; then
                        kill -KILL "$p" 2>/dev/null
                    fi
                done
                killed=true
            fi
        fi
    fi

    # 等待生效并验证
    sleep 1

    # 验证：使用完整命令检查
    if [ -n "$saved_command" ]; then
        local remaining=$(pgrep -f "$saved_command")
        if [ -n "$remaining" ]; then
            echo "错误: 终止失败，仍有进程运行: $remaining"
            return 1
        fi
    fi

    if [ "$killed" = true ]; then
        echo "服务已被强制终止"
        return 0
    else
        echo "未找到运行中的进程"
        return 0
    fi
}

# 重启服务
restart() {
    echo "重启 $SERVICE_NAME..."

    # 尝试stop，如果失败则自动使用kill
    if ! stop; then
        echo "stop失败，自动使用kill命令..."
        if ! kill_service; then
            echo "错误: kill也失败，请手动检查"
            return 1
        fi
    fi

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

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $(get_service_command)
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

# 显示日志
logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "错误: 日志文件不存在: $LOG_FILE"
        return 1
    fi

    echo "正在查看日志文件: $LOG_FILE"
    echo "按 Ctrl+C 退出"
    echo "========================================"
    tail -f "$LOG_FILE"
}

# 显示使用方法
usage() {
    echo "用法: $0 {start|stop|restart|debug|kill|status|logs} [cuda_device]"
    echo ""
    echo "命令:"
    echo "  start   - 后台启动服务"
    echo "  stop    - 停止服务（优雅关闭）"
    echo "  kill    - 强制终止服务（双重保险：PID+命令匹配）"
    echo "  restart - 重启服务（stop失败会自动使用kill）"
    echo "  debug   - 调试模式(前台运行)"
    echo "  status  - 显示运行状态"
    echo "  logs    - 查看实时日志"
    echo ""
    echo "参数:"
    echo "  cuda_device - CUDA设备号 (默认: $DEFAULT_CUDA_DEVICE)"
    echo ""
    echo "示例:"
    echo "  $0 start      # 使用默认CUDA设备启动"
    echo "  $0 start 0    # 使用CUDA设备0启动"
    echo "  $0 debug 1    # 使用CUDA设备1调试模式启动"
    echo ""
    echo "文件管理说明:"
    echo "  - PID/INFO文件只在start时创建和清理"
    echo "  - stop和kill命令不会删除这些文件"
    echo "  - 残留文件会在下次启动时自动清理"
}

# 主程序
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    kill)
        kill_service
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
    logs)
        logs
        ;;
    *)
        usage
        exit 1
        ;;
esac

exit $?