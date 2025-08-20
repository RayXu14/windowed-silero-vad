# @Time    : 2025/8/20 15:12
# @File    : speaker_verification.py
# @Version : 1.0
# @Description: 声纹识别服务 - 封装版本

import os
import time
import pathlib
import numpy as np
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import importlib
import soundfile as sf
import traceback
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger
import sys

# 日志配置
logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="INFO", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


# 配置参数
@dataclass
class SystemConfig:
    # 模型配置
    model_name: str = "speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
    model_name_alt: str = "speech_eres2net_large_200k_sv_zh-cn_16k-common"

    # 文件路径
    speaker_dir: str = "speaker"
    speakers_json: str = "speaker/speakers.json"
    embeddings_dir: str = "speaker/embeddings"

    # 音频配置
    sample_rate: int = 16000

    # 阈值配置
    sv_threshold: float = 0.55           # 声纹验证阈值
    min_register_len_sec: float = 1.0   # 最小注册音频长度(秒)


# 模型配置字典
MODEL_CONFIGS = {
    'speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'obj': 'eres2net.ERes2Net.ERes2Net',
        'args': {'feat_dim': 80, 'embedding_size': 512, 'm_channels': 64},
        'model_file': 'eres2net_large_model.ckpt',
        'model_path': "/root/.cache/modelscope/hub/models/iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
        },
    'speech_eres2net_large_200k_sv_zh-cn_16k-common': {
        'obj': 'eres2net.ERes2Net.ERes2Net',
        'args': {'feat_dim': 80, 'embedding_size': 512, 'm_channels': 64},
        'model_file': 'pretrained_eres2net.pt',
        'model_path': "/root/.cache/modelscope/hub/models/iic/speech_eres2net_large_200k_sv_zh-cn_16k-common"
        }
    }


class FBank(object):
    """特征提取器 - 优化版本"""

    def __init__(self, n_mels, sample_rate, mean_nor: bool = False):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)

        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
                           sample_frequency=self.sample_rate, dither=dither)
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class EmbeddingManager:
    """Embedding管理器 - 核心优化组件"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.device = None
        self.feature_extractor = FBank(80, sample_rate=config.sample_rate, mean_nor=True)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.speaker_metadata: Dict[str, dict] = {}

        # 确保目录存在
        pathlib.Path(config.speaker_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(config.embeddings_dir).mkdir(exist_ok=True, parents=True)

        self._load_model()
        self._load_speakers_cache()

    def _dynamic_import(self, import_path):
        """动态导入模块"""
        module_name, obj_name = import_path.rsplit('.', 1)
        m = importlib.import_module(module_name)
        return getattr(m, obj_name)

    def _load_model(self):
        """加载embedding模型"""
        start_time = time.time()
        logger.info(f"[模型] 加载模型: {self.config.model_name_alt}")

        model_config = MODEL_CONFIGS[self.config.model_name_alt]
        model_path = pathlib.Path(model_config['model_path'])

        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        pretrained_model = model_path / model_config['model_file']
        pretrained_state = torch.load(pretrained_model, map_location='cpu')

        self.model = self._dynamic_import(model_config['obj'])(**model_config['args'])
        self.model.load_state_dict(pretrained_state)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        load_time = (time.time() - start_time) * 1000
        logger.info(f"[模型] 加载完成，耗时: {load_time:.1f}ms，设备: {self.device}")

    def _load_speakers_cache(self):
        """加载说话人缓存"""
        start_time = time.time()

        if not os.path.exists(self.config.speakers_json):
            logger.info("[缓存] speakers.json不存在，初始化空缓存")
            return

        with open(self.config.speakers_json, 'r', encoding='utf-8') as f:
            self.speaker_metadata = json.load(f)

        # 加载embedding缓存
        loaded_count = 0
        generated_count = 0
        metadata_updated = False

        for speaker_id, metadata in self.speaker_metadata.items():
            embedding_path = metadata.get('embedding_path')

            # 检查embedding是否存在
            if embedding_path and os.path.exists(embedding_path):
                # 直接加载现有embedding
                try:
                    self.embedding_cache[speaker_id] = np.load(embedding_path)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"[缓存] 加载{speaker_id}的embedding失败: {e}")
            else:
                # 缺少embedding，尝试从音频文件生成
                audio_path = metadata.get('path')
                if audio_path and os.path.exists(audio_path):
                    try:
                        logger.info(f"[自动生成] 为{speaker_id}生成embedding...")

                        # 读取音频文件
                        audio, sr = sf.read(audio_path)
                        if sr != self.config.sample_rate:
                            # 重采样（需要安装scipy）
                            import scipy.signal
                            audio = scipy.signal.resample(audio, int(len(audio) * self.config.sample_rate / sr))

                        # 生成embedding
                        embedding = self.generate_embedding_optimized(audio)

                        # 保存embedding文件
                        embedding_path = os.path.join(self.config.embeddings_dir, f"{speaker_id}.npy")
                        np.save(embedding_path, embedding)

                        # 更新缓存和元数据
                        self.embedding_cache[speaker_id] = embedding
                        self.speaker_metadata[speaker_id]['embedding_path'] = embedding_path
                        self.speaker_metadata[speaker_id]['sr'] = self.config.sample_rate

                        generated_count += 1
                        metadata_updated = True
                        logger.info(f"[自动生成] {speaker_id}的embedding生成完成")

                    except Exception as e:
                        logger.error(f"[自动生成] 为{speaker_id}生成embedding失败: {e}")
                else:
                    logger.warning(f"[缺失] {speaker_id}的音频文件不存在: {audio_path}")

        # 如果有更新，保存元数据
        if metadata_updated:
            self._save_speakers_metadata()
            logger.info(f"[元数据] 已更新speakers.json，新增{generated_count}个embedding")

        load_time = (time.time() - start_time) * 1000
        logger.info(f"[缓存] 加载完成，说话人: {len(self.speaker_metadata)}个，embedding: {loaded_count+generated_count}个，耗时: {load_time:.1f}ms")

    def _save_speakers_metadata(self):
        """保存说话人元数据"""
        with open(self.config.speakers_json, 'w', encoding='utf-8') as f:
            json.dump(self.speaker_metadata, f, indent=2, ensure_ascii=False)

    def generate_embedding_optimized(self, audio: np.ndarray) -> np.ndarray:
        """优化的embedding生成"""
        start_time = time.time()

        # 音频预处理优化
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)

        # 长度处理优化
        max_length = int(90 * self.config.sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]

        # 分块处理优化
        chunk_size = int(10 * self.config.sample_rate)

        if len(audio) < chunk_size:
            # 短音频：零填充而非循环填充
            padded_audio = np.zeros(chunk_size, dtype=audio.dtype)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
            chunks = [audio]
        else:
            # 长音频：重叠分块
            chunks = []
            step = chunk_size // 2  # 50%重叠
            for i in range(0, len(audio) - chunk_size + 1, step):
                chunks.append(audio[i:i + chunk_size])

            # 确保最后一块
            if len(audio) % step != 0:
                last_chunk = audio[-chunk_size:]
                chunks.append(last_chunk)

        # 批量特征提取
        feat_start = time.time()
        feats = []
        for chunk in chunks:
            feat = self.feature_extractor(torch.from_numpy(chunk).float().unsqueeze(0))
            feats.append(feat)

        if feats:
            feats_tensor = torch.stack(feats).float().to(self.device)
        else:
            # 空音频处理
            dummy_chunk = np.zeros(chunk_size, dtype=np.float32)
            feat = self.feature_extractor(torch.from_numpy(dummy_chunk).unsqueeze(0))
            feats_tensor = feat.unsqueeze(0).to(self.device)

        feat_time = (time.time() - feat_start) * 1000

        # 模型推理
        infer_start = time.time()
        with torch.no_grad():
            embeddings = self.model(feats_tensor).cpu().numpy()
            final_embedding = embeddings.mean(0)  # 平均池化

        infer_time = (time.time() - infer_start) * 1000
        total_time = (time.time() - start_time) * 1000

        logger.info(f"[embedding] 特征: {feat_time:.1f}ms, 推理: {infer_time:.1f}ms, 总计: {total_time:.1f}ms")
        return final_embedding

    def cosine_similarity_batch(self, target_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """批量余弦相似度计算"""
        if not self.embedding_cache:
            return []

        start_time = time.time()

        # 向量化计算
        speaker_ids = list(self.embedding_cache.keys())
        cached_embeddings = np.stack([self.embedding_cache[sid] for sid in speaker_ids])

        # 批量余弦相似度
        target_norm = np.linalg.norm(target_embedding)
        cached_norms = np.linalg.norm(cached_embeddings, axis=1)

        similarities = np.dot(cached_embeddings, target_embedding) / (cached_norms * target_norm)

        results = [(speaker_ids[i], float(similarities[i])) for i in range(len(speaker_ids))]

        calc_time = (time.time() - start_time) * 1000
        logger.info(f"[相似度] 批量计算{len(results)}个，耗时: {calc_time:.1f}ms")

        return results

    def verify_speaker(self, audio: np.ndarray, threshold: float = None) -> Tuple[bool, Optional[str], float]:
        """说话人验证 - 主要接口"""
        if threshold is None:
            threshold = self.config.sv_threshold

        start_time = time.time()

        # 生成目标embedding
        target_embedding = self.generate_embedding_optimized(audio)

        # 批量计算相似度
        similarities = self.cosine_similarity_batch(target_embedding)

        if not similarities:
            total_time = (time.time() - start_time) * 1000
            logger.info(f"[验证] 无注册说话人，耗时: {total_time:.1f}ms")
            return False, None, 0.0

        # 找到最高相似度
        best_speaker, best_score = max(similarities, key=lambda x: x[1])
        hit = best_score >= threshold

        total_time = (time.time() - start_time) * 1000
        logger.info(f"[验证] 最佳匹配: {best_speaker}({best_score:.4f}), 命中: {hit}, 耗时: {total_time:.1f}ms")

        return hit, best_speaker if hit else None, best_score

    def register_speaker(self, audio: np.ndarray, speaker_id: str = None) -> Tuple[str, bool]:
        """注册新说话人"""
        start_time = time.time()

        # 自动生成ID
        if speaker_id is None:
            existing_ids = [k for k in self.speaker_metadata.keys() if k.startswith("speaker_")]
            max_n = max([int(k.split('_')[1]) for k in existing_ids
                         if '_' in k and len(k.split('_')) > 1 and k.split('_')[1].isdigit()] + [0])
            speaker_id = f"speaker_{max_n + 1}_{int(time.time())}"

        # 保存音频文件
        audio_path = os.path.join(self.config.speaker_dir, f"{speaker_id}.wav")
        sf.write(audio_path, audio, self.config.sample_rate)

        # 生成embedding
        embedding = self.generate_embedding_optimized(audio)

        # 保存embedding文件
        embedding_path = os.path.join(self.config.embeddings_dir, f"{speaker_id}.npy")
        np.save(embedding_path, embedding)

        # 更新缓存
        self.embedding_cache[speaker_id] = embedding
        self.speaker_metadata[speaker_id] = {
            "path": audio_path,
            "sr": self.config.sample_rate,
            "embedding_path": embedding_path
            }

        # 保存元数据
        self._save_speakers_metadata()

        reg_time = (time.time() - start_time) * 1000
        logger.info(f"[注册] 新说话人: {speaker_id}, 耗时: {reg_time:.1f}ms")

        return speaker_id, True

    def verify_and_register(self, audio: np.ndarray, threshold: float = None,
                            min_length_sec: float = None) -> Tuple[bool, Optional[str], bool]:
        """验证并自动注册 - 集成接口"""
        if threshold is None:
            threshold = self.config.sv_threshold
        if min_length_sec is None:
            min_length_sec = self.config.min_register_len_sec

        # 首先尝试验证
        hit, speaker_id, score = self.verify_speaker(audio, threshold)

        if hit:
            return True, speaker_id, False

        # 检查音频长度
        audio_len_sec = len(audio) / self.config.sample_rate
        if audio_len_sec < min_length_sec:
            logger.info(f"[自动注册] 音频太短({audio_len_sec:.1f}s < {min_length_sec}s)，跳过注册")
            return False, None, False

        # 自动注册
        new_speaker_id, success = self.register_speaker(audio)
        return success, new_speaker_id, True


# ===================== 封装接口 =====================

class SpeakerVerificationService:
    """声纹识别服务 - 简化接口"""

    def __init__(self, threshold: float = 0.1, min_register_length: float = 1.0):
        """
        初始化声纹识别服务

        Args:
            threshold: 相似度阈值，默认0.1
            min_register_length: 最小注册音频长度(秒)，默认1.0
        """
        self.config = SystemConfig()
        self.config.sv_threshold = threshold
        self.config.min_register_len_sec = min_register_length

        self.embedding_manager = EmbeddingManager(self.config)

    def identify_speaker(self, audio: np.ndarray) -> dict:
        """
        识别说话人（验证+自动注册）

        Args:
            audio: 音频数据(numpy数组)

        Returns:
            dict: {
                'speaker_id': str,           # 说话人ID
                'is_registered': bool,       # 是否为已注册说话人
                'is_new_registration': bool, # 是否为新注册
                'confidence': float          # 相似度分数
            }
        """
        hit, speaker_id, is_new = self.embedding_manager.verify_and_register(audio)

        if hit and speaker_id:
            # 获取相似度分数
            _, _, confidence = self.embedding_manager.verify_speaker(audio)
            return {
                'speaker_id': speaker_id,
                'is_registered': True,
                'is_new_registration': is_new,
                'confidence': confidence
                }
        else:
            return {
                'speaker_id': None,
                'is_registered': False,
                'is_new_registration': False,
                'confidence': 0.0
                }

    def register_speaker(self, audio: np.ndarray, speaker_id: str = None) -> dict:
        """
        显式注册说话人

        Args:
            audio: 音频数据
            speaker_id: 指定的说话人ID，None则自动生成

        Returns:
            dict: {
                'speaker_id': str,  # 分配的说话人ID
                'success': bool     # 是否注册成功
            }
        """
        speaker_id, success = self.embedding_manager.register_speaker(audio, speaker_id)
        return {
            'speaker_id': speaker_id,
            'success': success
            }

    def verify_speaker(self, audio: np.ndarray) -> dict:
        """
        仅验证说话人（不自动注册）

        Args:
            audio: 音频数据

        Returns:
            dict: {
                'speaker_id': str,      # 匹配的说话人ID
                'is_match': bool,       # 是否匹配成功
                'confidence': float     # 相似度分数
            }
        """
        hit, speaker_id, confidence = self.embedding_manager.verify_speaker(audio)
        return {
            'speaker_id': speaker_id,
            'is_match': hit,
            'confidence': confidence
            }

    def get_registered_speakers(self) -> List[str]:
        """获取所有已注册的说话人ID列表"""
        return list(self.embedding_manager.speaker_metadata.keys())

    def get_speaker_count(self) -> int:
        """获取已注册说话人数量"""
        return len(self.embedding_manager.speaker_metadata)


# ===================== 使用示例 =====================

def main():
    """使用示例"""
    # 方式1: 直接使用原始类
    config = SystemConfig()
    embedding_manager = EmbeddingManager(config)

    # 方式2: 使用封装的服务类
    speaker_service = SpeakerVerificationService(threshold=0.1, min_register_length=1.0)

    # 示例音频数据（实际使用时从文件加载）
    # audio_data = ...

    # 识别说话人
    # result = speaker_service.identify_speaker(audio_data)
    # print(f"识别结果: {result}")


if __name__ == "__main__":
    main()
