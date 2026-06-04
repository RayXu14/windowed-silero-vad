# windowed-silero-vad (EMA + early-onset + tail-extend)

基于 Silero VAD 的流式语音端点检测 WebSocket 服务，检测到完整语音段后切片送 ASR 转写。

本分支是线上原版 windowed-silero-vad 的**增强版**：原有 `audio_chunk` 上行 + `vad`/`asr` 下行协议**完全保留**，不发握手的老客户端无需任何改动。新增**可选**的 `session.init` 握手（见下方「按连接配置」），让每条连接能传入自己的 VAD 参数。在原版滑窗均值平滑的基础上增加三项增强，目标是降低误触发、找回被切掉的首尾音频：

- **EMA 概率平滑**：指数平滑替代滑窗均值，抑制概率抖动导致的误触发。
- **early-onset 起点回溯**：状态机确认进入语音后，用原始(未平滑)概率向前回溯，把被平滑/阈值切掉的前导字补回语音段。
- **tail-extend 段尾延伸**：段结束前用原始概率向后延伸，找回被切掉的尾音（early-onset 的镜像）。

三项增强均可独立开关，关闭后行为与线上原版一致。

## 安装依赖

```bash
uv sync
```

## 启动

需先有一个 ASR 服务（兼容 `POST /asr/transcribe`）。下面三种启动方式按需选用。

### 方式一：app.sh（推荐，带进程管理 + 参数集中）

所有可调参数集中在 app.sh 顶部「配置区」，每项都可用**同名环境变量覆盖**。

```bash
bash app.sh start                              # 默认参数后台启动
bash app.sh start 50162 http://localhost:50300 # 指定端口和 ASR 地址
PROB_THRESHOLD=0.5 TAIL_FLOOR=0.1 bash app.sh start  # 环境变量覆盖默认参数
bash app.sh status                             # 查看运行状态
bash app.sh logs                               # 跟随实时日志
bash app.sh stop                               # 优雅停止
bash app.sh restart                            # 重启(stop 失败自动 kill)
bash app.sh debug                              # 前台调试运行
```

子命令：`start | stop | kill | restart | debug | status | logs`。

### 方式二：run_server.sh（简易，含平滑预设）

预设 G2(win=8)/G1(win=6) 与原版平滑对照，默认端口 50161。

```bash
bash run_server.sh                       # G2: win=8 + ema
SMOOTH_WIN=6 bash run_server.sh          # G1: win=6 + ema
SMOOTH_METHOD=mean bash run_server.sh    # 退回原版滑窗均值平滑(对照)
```

> 注意：run_server.sh 内的 venv 路径为线上服务器绝对路径，本机运行请改用 app.sh 或下面的 uv run。

### 方式三：裸 uv run（手工传全部参数）

```bash
uv run vad_websocket_server.py --asr-url http://localhost:50300 --port 50162 \
  --smoothing-method ema --smoothing-window 6 \
  --prob-threshold 0.4 --exit-prob-threshold 0.25 \
  --enable-early-onset --enable-tail-extend
```

## 测试

```bash
uv run test_realtime_client.py            # 实时麦克风
uv run test_client.py /path/to/wav/dir    # 离线 WAV 目录
uv run test_ws_init.py                    # session.init 握手协议四路
```

## 按连接配置（session.init）

每条 WS 连接各自持有一个独立的 VAD 处理器。默认所有连接用服务启动时的全局参数；客户端可在**连接后的首帧**发一条 `session.init` 覆盖本连接的 VAD 参数，互不影响。

**上行 `session.init`**（首帧，可选）：

```json
{ "type": "session.init", "config": { "prob_threshold": 0.6, "enable_early_onset": true } }
```

- `config` 里只放想覆盖的字段，未给的字段取全局默认。可覆盖字段 = 下方「关键参数」表中的全部 VAD 调优项（`asr-url`、端口等部署级参数不开放）。
- 字段名用**下划线**形式（如 `prob_threshold`、`enable_tail_extend`），与命令行的连字符形式对应。

**下行 `ready`**：服务端建好处理器后回 `ready`，并带回本连接**实际生效**的全部参数供核对：

```json
{ "type": "status", "status": "ready", "message": "...", "effective_config": { ... } }
```

**错误处理**：`config` 含未知字段、类型不符、或违反约束（如 `exit_prob_threshold > prob_threshold`、`prebuffer + required_hits > required_misses`）时，回 `{ "type": "error", "error": "session.init 配置无效: ..." }` 并**关闭连接**。

**兼容性**：`session.init` 可选。

- 不发 init、首帧直接是 `audio_chunk` → 用全局默认建处理器，正常工作。
- 老客户端连上后等 `ready` 才发音频 → 服务端等首帧最多 1 秒，超时即用全局默认建处理器并发 `ready`（老客户端零改动）。

## 关键参数

完整默认值见 app.sh 配置区；命令行参数同名（连字符形式）。

| 类别 | 参数 | 默认 | 说明 |
|---|---|---|---|
| 概率平滑 | `--smoothing-method` | ema | `ema` 指数平滑(增强) / `mean` 滑窗均值(原版) |
| | `--smoothing-window` | 6 | 平滑窗口 |
| 核心门控 | `--prob-threshold` | 0.4 | IDLE→ACTIVE 进入阈值(平滑概率) |
| | `--exit-prob-threshold` | 0.25 | ACTIVE 维持阈值(低于进入阈值，保尾音) |
| | `--volume-threshold` | -30 | 最低音量门限(dB) |
| | `--snr-margin-db` | 28 | IDLE 进入需高于噪声底的 dB |
| | `--noise-floor-window` | 120 | 噪声底估计窗(chunk 数) |
| 状态机 | `--required-hits` | 3 | 连续命中几次进 ACTIVE |
| | `--required-misses` | 20 | 连续未命中几次退 IDLE |
| | `--prebuffer` | 16 | 进 ACTIVE 时带的前置 chunk 数 |
| early-onset | `--enable-early-onset` | (app.sh 默认开) | 起点回溯开关 |
| | `--early-onset-floor` | 0.01 | 回溯判定"仍在语音"的原始概率地板 |
| | `--early-onset-release` | 10 | 连续低于地板多少帧停止回溯 |
| | `--early-onset-max-lookback` | 20 | 额外往前回溯的最大帧数 |
| tail-extend | `--enable-tail-extend` | (app.sh 默认开) | 段尾延伸开关 |
| | `--tail-floor` | 0.01 | 段尾延伸判定"仍是尾音"的原始概率地板(需 < exit 阈值) |
| | `--tail-release` | 10 | 连续低于地板多少帧停止延伸 |
| | `--tail-max-lookahead` | 20 | 段尾最多延伸帧数(≤ required-misses) |

> 说明：early-onset / tail-extend 仅向语音段补回音频喂给 ASR，对外下发的语音**起点时间戳维持基线判定时刻**，不随回溯前移。
