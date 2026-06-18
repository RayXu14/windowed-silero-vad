#!/bin/bash

# windowed-silero-vad 启动脚本
# 用法: bash vad.sh {start|stop|restart|debug|kill|status|logs} [port] [asr-url]

# ============ 配置区域 ============
# 所有可调参数集中在此。每项都可用同名环境变量覆盖,例如:
#   PROB_THRESHOLD=0.5 TAIL_FLOOR=0.1 bash app.sh start
SERVICE_NAME="windowed-silero-vad-ema"  # 服务名称
RUNTIME_DIR="logs"                  # 运行时文件目录
PYTHON_CMD="uv run"                 # 用本目录独立venv
SCRIPT_NAME="vad_websocket_server.py"  # 主脚本文件名
PORT=${2:-50160}                    # 服务端口
ASR_URL=${3:-"http://localhost:50300"}  # ASR服务地址

# ---- 概率平滑 ----
SMOOTH_WIN=${SMOOTH_WIN:-6}                  # 平滑窗口(G2=8, G1=6)
SMOOTH_METHOD=${SMOOTH_METHOD:-ema}          # 平滑方式(ema=改进, mean=线上原版)

# ---- VAD 核心门控 ----
PROB_THRESHOLD=${PROB_THRESHOLD:-0.4}        # IDLE→ACTIVE 进入阈值(平滑概率)
EXIT_PROB_THRESHOLD=${EXIT_PROB_THRESHOLD:-0.25}  # ACTIVE 维持阈值(<进入,保尾音)
VOLUME_THRESHOLD=${VOLUME_THRESHOLD:--48}    # 最低音量门限(dB)
SNR_MARGIN_DB=${SNR_MARGIN_DB:-28}           # IDLE 进入需高于噪声底的 dB
ACTIVE_SNR_MARGIN_DB=${ACTIVE_SNR_MARGIN_DB:-0}   # ACTIVE 维持需高于噪声底的 dB
NOISE_FLOOR_WINDOW=${NOISE_FLOOR_WINDOW:-120}     # 噪声底估计窗(chunk数)
NOISE_UPDATE_PROB=${NOISE_UPDATE_PROB:-0.5}  # 仅平滑概率低于此值才更新噪声底

# ---- 状态机 ----
REQUIRED_HITS=${REQUIRED_HITS:-3}            # 连续命中几次进 ACTIVE
REQUIRED_MISSES=${REQUIRED_MISSES:-20}       # 连续未命中几次退 IDLE
PREBUFFER=${PREBUFFER:-16}                   # 进 ACTIVE 时带的前置 chunk 数

# ---- early-onset 起点回溯(段头找回前导字) ----
EARLY_ONSET=${EARLY_ONSET:-"--enable-early-onset"}   # 置空 "" 则关闭
EARLY_ONSET_FLOOR=${EARLY_ONSET_FLOOR:-0.01}         # 回溯地板概率
EARLY_ONSET_RELEASE=${EARLY_ONSET_RELEASE:-10}        # 连续低于地板多少帧停止回溯
EARLY_ONSET_LOOKBACK=${EARLY_ONSET_LOOKBACK:-20}     # 额外往前回溯的最大帧数

# ---- tail-extend 段尾延伸(段尾找回尾音) ----
TAIL_EXTEND=${TAIL_EXTEND:-"--enable-tail-extend"}   # 置空 "" 则关闭
TAIL_FLOOR=${TAIL_FLOOR:-0.00}                       # 段尾地板概率(需<EXIT_PROB_THRESHOLD才有效)
TAIL_RELEASE=${TAIL_RELEASE:-10}                      # 连续低于地板多少帧停止延伸
TAIL_LOOKAHEAD=${TAIL_LOOKAHEAD:-20}                 # 最多延伸帧数(<=REQUIRED_MISSES)
# ================================

PID_FILE="$RUNTIME_DIR/app.pid"
INFO_FILE="$RUNTIME_DIR/app.info"
LOG_FILE="$RUNTIME_DIR/app.log"

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

# 构建完整的启动命令(参数全部取自上方配置区变量)
get_app_command() {
    echo "$PYTHON_CMD $SCRIPT_NAME --port $PORT --asr-url $ASR_URL \
--prob-threshold $PROB_THRESHOLD --exit-prob-threshold $EXIT_PROB_THRESHOLD \
--volume-threshold $VOLUME_THRESHOLD --snr-margin-db $SNR_MARGIN_DB \
--active-snr-margin-db $ACTIVE_SNR_MARGIN_DB --noise-floor-window $NOISE_FLOOR_WINDOW \
--noise-update-prob-threshold $NOISE_UPDATE_PROB \
--required-hits $REQUIRED_HITS --required-misses $REQUIRED_MISSES --prebuffer $PREBUFFER \
--smoothing-window $SMOOTH_WIN --smoothing-method $SMOOTH_METHOD \
$EARLY_ONSET --early-onset-floor $EARLY_ONSET_FLOOR --early-onset-release $EARLY_ONSET_RELEASE --early-onset-max-lookback $EARLY_ONSET_LOOKBACK \
$TAIL_EXTEND --tail-floor $TAIL_FLOOR --tail-release $TAIL_RELEASE --tail-max-lookahead $TAIL_LOOKAHEAD"
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

    echo "启动 $SERVICE_NAME (端口: $PORT, ASR地址: $ASR_URL)..."
    echo "使用命令: $(get_app_command)"

    nohup $(get_app_command) > $LOG_FILE 2>&1 & echo $! > $PID_FILE

    local pid=$(cat $PID_FILE)

    # 保存完整启动命令和启动时间到INFO文件
    cat > "$INFO_FILE" << EOF
PID=$pid
COMMAND=$(get_app_command)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE=$LOG_FILE
PORT=$PORT
ASR_URL=$ASR_URL
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

    echo "$SERVICE_NAME 已停止"
    return 0
}

# 强制终止服务（双重保险：PID+完整命令匹配）
kill_service() {
    local killed=false
    local saved_command=""

    # 方案1：通过PID强制终止（先TERM再KILL）
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "通过PID终止进程 (PID: $pid)..."

            kill -TERM "$pid" 2>/dev/null

            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 3 ]; do
                sleep 1
                count=$((count + 1))
            done

            if ps -p "$pid" > /dev/null 2>&1; then
                echo "进程未响应TERM信号，强制终止..."
                kill -KILL "$pid" 2>/dev/null
            fi

            killed=true
        fi
    fi

    # 方案2：通过完整命令匹配（备用方案）
    if [ -f "$INFO_FILE" ]; then
        saved_command=$(grep "^COMMAND=" "$INFO_FILE" | cut -d'=' -f2-)
        if [ -n "$saved_command" ]; then
            echo "使用完整命令匹配查找进程..."
            local pids=$(pgrep -f "$saved_command")

            if [ -n "$pids" ]; then
                echo "找到匹配的进程: $pids"
                echo "$pids" | xargs kill -TERM 2>/dev/null
                sleep 2
                for p in $pids; do
                    if ps -p "$p" > /dev/null 2>&1; then
                        kill -KILL "$p" 2>/dev/null
                    fi
                done
                killed=true
            fi
        fi
    fi

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

    echo "调试模式启动 $SERVICE_NAME (端口: $PORT, ASR地址: $ASR_URL)..."
    echo "使用命令: $(get_app_command)"
    echo "按 Ctrl+C 退出"

    $(get_app_command)
}

# 显示状态
status() {
    if is_running; then
        PID=$(cat $PID_FILE)
        echo "$SERVICE_NAME 正在运行 (PID: $PID, 端口: $PORT)"
        echo "ASR地址: $ASR_URL"
        echo "日志文件: $LOG_FILE"
        echo "查看实时日志: tail -f $LOG_FILE"
    else
        echo "$SERVICE_NAME 未在运行"
    fi
}

# 查看日志
logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "日志文件不存在: $LOG_FILE"
        return 1
    fi

    echo "查看 $SERVICE_NAME 日志 (按 Ctrl+C 退出):"
    echo "日志文件: $LOG_FILE"
    echo "----------------------------------------"
    tail -f "$LOG_FILE"
}

# 显示使用方法
usage() {
    echo "用法: $0 {start|stop|restart|debug|kill|status|logs} [port] [asr-url]"
    echo ""
    echo "命令:"
    echo "  start   - 后台启动服务"
    echo "  stop    - 停止服务（优雅关闭）"
    echo "  kill    - 强制终止服务（双重保险：PID+命令匹配）"
    echo "  restart - 重启服务（stop失败会自动使用kill）"
    echo "  debug   - 调试模式（前台运行）"
    echo "  status  - 显示运行状态"
    echo "  logs    - 查看实时日志"
    echo ""
    echo "配置:"
    echo "  脚本文件: $SCRIPT_NAME"
    echo "  服务端口: $PORT"
    echo "  ASR地址:  $ASR_URL"
    echo ""
    echo "示例:"
    echo "  $0 start                                      # 使用默认参数启动"
    echo "  $0 start 50160                                # 指定端口"
    echo "  $0 start 50160 http://localhost:50300         # 指定端口和ASR地址"
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
