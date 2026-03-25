"""
SenseVoice ASR REST API 服务

接口规范：
    POST /asr/transcribe
    请求：{ "audio_base64": "...", "format": "wav", "language": "zh" }
    响应：{ "results": [{"text": "...", "language": "zh", "emotion": "NEUTRAL", "event": "Speech", "itn": true}], "elapsed_ms": 310.2 }
"""

import base64
import io
import re
import time

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
import uvicorn

from asr import asr, format_str_v3


app = FastAPI(title="SenseVoice ASR Server", version="1.0.0")

SAMPLING_RATE = 16000


class TranscribeRequest(BaseModel):
    audio_base64: str
    format: str = "wav"
    language: str | None = None
    terms: list[str] | None = None
    system: str | None = None


class TranscribeResultItem(BaseModel):
    text: str
    language: str
    emotion: str
    event: str
    itn: bool


class TranscribeResponse(BaseModel):
    results: list[TranscribeResultItem]
    elapsed_ms: float


def decode_audio_from_base64(audio_b64: str, audio_format: str) -> torch.Tensor:
    """将 base64 编码的音频解码为 16kHz 单声道 float32 torch tensor"""
    # 去除可能的 data URL 前缀
    if "," in audio_b64 and audio_b64.startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_b64)
    buffer = io.BytesIO(audio_bytes)

    waveform, sample_rate = torchaudio.load(buffer, format=audio_format)

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样到 16kHz
    if sample_rate != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE)
        waveform = resampler(waveform)

    # 返回 (samples,) 形状的 tensor，与原 asr() 函数输入一致
    return waveform.squeeze(0)


@app.post("/asr/transcribe", response_model=TranscribeResponse)
def transcribe(request: TranscribeRequest):
    try:
        audio_tensor = decode_audio_from_base64(request.audio_base64, request.format)
    except Exception as e:
        logger.opt(exception=True).error("音频解码失败")
        raise HTTPException(status_code=400, detail=f"音频解码失败: {e}")

    duration = len(audio_tensor) / SAMPLING_RATE
    if duration > 120:
        raise HTTPException(status_code=400, detail=f"音频时长 {duration:.1f}s 超过 120s 限制")

    lang = request.language or "auto"

    start_ms = time.time()
    try:
        # 每次请求使用独立的 cache
        result = asr(audio_tensor, lang, cache={}, use_itn=True)
    except Exception as e:
        logger.opt(exception=True).error("ASR 识别失败")
        raise HTTPException(status_code=503, detail=f"ASR 识别失败: {e}")
    elapsed_ms = (time.time() - start_ms) * 1000

    raw_text = result[0]["text"]
    cleaned_text = format_str_v3(raw_text)

    # 解析 SenseVoice 结构化标签: <|lang|><|emotion|><|event|><|itn|>text
    tags = re.findall(r"<\|([^|]+)\|>", raw_text)

    LANGUAGES = {"zh", "en", "ja", "ko", "yue", "nospeech"}
    EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
    EVENTS = {"BGM", "Speech", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough", "Sing", "Speech_Noise", "Event_UNK"}

    detected_lang = lang
    detected_emotion = "NEUTRAL"
    detected_event = "Speech"
    detected_itn = False

    for tag in tags:
        if tag in LANGUAGES:
            detected_lang = tag
        elif tag in EMOTIONS:
            detected_emotion = tag
        elif tag in EVENTS:
            detected_event = tag
        elif tag == "withitn":
            detected_itn = True
        elif tag == "woitn":
            detected_itn = False

    return TranscribeResponse(
        results=[TranscribeResultItem(
            text=cleaned_text,
            language=detected_lang,
            emotion=detected_emotion,
            event=detected_event,
            itn=detected_itn,
        )],
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get("/")
def root():
    return {
        "service": "SenseVoice ASR Server",
        "endpoint": "POST /asr/transcribe",
        "audio_specs": {
            "sampling_rate": f"{SAMPLING_RATE} Hz",
            "supported_formats": ["wav", "mp3", "flac"],
            "max_duration_seconds": 120,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SenseVoice ASR REST API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50300)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "TRACE"])

    args = parser.parse_args()

    logger.add("asr_server.log", level=args.log_level)

    logger.info(f"ASR 服务启动: http://{args.host}:{args.port}")
    logger.info(f"接口地址: http://{args.host}:{args.port}/asr/transcribe")

    uvicorn.run(app, host=args.host, port=args.port)
