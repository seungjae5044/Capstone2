"""
FastAPI 기반 Whisper 실시간 전사 웹 UI
MeetingProgram의 UI를 재사용하여 test_whisper2.py 전사 결과와 Ollama 평가를 표출한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from math import gcd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import requests
import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket 연결 관리자"""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logger.info("WebSocket 연결 추가: %s개 활성", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("WebSocket 연결 해제: %s개 활성", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        async with self.lock:
            connections = list(self.active_connections)
        if not connections:
            return
        disconnect_list: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as exc:  # noqa: BLE001
                logger.error("WebSocket 전송 실패: %s", exc)
                disconnect_list.append(connection)
        if disconnect_list:
            async with self.lock:
                for connection in disconnect_list:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)


app = FastAPI(
    title="Whisper Meeting UI",
    description="Whisper 실시간 전사 + Ollama 평가 대시보드",
    version="1.0.0",
)
manager = ConnectionManager()


DEFAULT_CONFIG: Dict[str, Any] = {
    "model_id": "openai/whisper-large-v3-turbo",
    "language": "ko",
    "force_device": None,
    "audio_source_sr": 48000,
    "target_sr": 16000,
    "chunk_length_s": 5,
    "stride_seconds": 0.8,
    "chunk_duration": 5.0,
    "blocksize_seconds": 0.2,
    "queue_maxsize": 100,
    "batch_size": 1,
    "silence_rms_threshold": 0.02,
    "generate_kwargs": {
        "task": "transcribe",
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "repetition_penalty": 1.2,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """중첩 딕셔너리를 병합한다."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    """구성 파일을 불러온다."""

    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config = deep_update(config, user_config)
    return config


def select_device(force_device: Optional[str]) -> tuple[str, torch.dtype]:
    """가용한 장치를 선택한다."""

    if force_device:
        requested = force_device.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda", torch.float16
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16
        if requested == "cpu":
            return "cpu", torch.float32
        logger.warning("요청한 장치 %s 를 사용할 수 없습니다. 자동 선택으로 진행합니다.", force_device)

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@dataclass
class WhisperConfig:
    model_id: str
    language: str
    audio_source_sr: int
    target_sr: int
    chunk_length_s: int
    stride_seconds: int
    chunk_duration: float
    blocksize_seconds: float
    queue_maxsize: int
    batch_size: int
    silence_rms_threshold: float
    generate_kwargs: Dict[str, Any]
    force_device: Optional[str] = None


@dataclass
class TranscribedSegment:
    text: str
    speaker_id: str
    speaker_name: str
    similarity: float
    start_time: datetime
    end_time: datetime
    duration: float


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    embedding: np.ndarray
    statement_count: int = 0
    duration: float = 0.0
    last_similarity: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_embedding(self, embedding: np.ndarray, alpha: float) -> None:
        if not np.any(self.embedding):
            self.embedding = embedding
        else:
            self.embedding = (1.0 - alpha) * self.embedding + alpha * embedding
        self.embedding = self.embedding / (np.linalg.norm(self.embedding) + 1e-8)


@dataclass
class TimelineSegment:
    speaker_id: str
    speaker_name: str
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": (self.end_time - self.start_time).total_seconds(),
        }


class SileroVAD:
    def __init__(self, threshold: float = 0.5, device: str = "cpu") -> None:
        self.threshold = threshold
        self.device = device
        self.model = None
        self.get_speech_ts = None
        self.lock = threading.Lock()

    def _initialize(self) -> None:
        with self.lock:
            if self.model is not None and self.get_speech_ts is not None:
                return
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.model = model.to("cpu")
            self.get_speech_ts = utils[0]

    def is_speech(self, audio: np.ndarray, sr: int) -> bool:
        if audio.size < max(1600, int(sr * 0.2)):
            return False
        if self.model is None or self.get_speech_ts is None:
            self._initialize()
        if self.model is None or self.get_speech_ts is None:
            return False
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        with torch.no_grad():
            speech_timestamps = self.get_speech_ts(
                audio_tensor,
                self.model,
                sampling_rate=sr,
                threshold=self.threshold,
                return_seconds=False,
            )
        return bool(speech_timestamps)


class SpeechBrainEncoder:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device if device == "cuda" else "cpu"
        self.encoder = None
        self.lock = threading.Lock()

    def _initialize(self) -> None:
        with self.lock:
            if self.encoder is not None:
                return
            from speechbrain.pretrained import EncoderClassifier

            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )

    def embed(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if audio.size == 0:
            return None
        if self.encoder is None:
            self._initialize()
        if self.encoder is None:
            return None
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            embedding = self.encoder.encode_batch(waveform)
        emb = embedding.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm


class SpeakerHandler:
    def __init__(
        self,
        max_speakers: int = 10,
        similarity_threshold: float = 0.35,
        update_alpha: float = 0.2,
    ) -> None:
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.update_alpha = update_alpha
        self._counter = 1
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._counter = 1
            self._profiles.clear()

    def classify(self, embedding: Optional[np.ndarray]) -> Tuple[str, SpeakerProfile, float]:
        if embedding is None or not np.any(embedding):
            return self._assign_default()

        with self._lock:
            best_id = None
            best_sim = -1.0
            for speaker_id, profile in self._profiles.items():
                similarity = float(np.dot(profile.embedding, embedding))
                if similarity > best_sim:
                    best_sim = similarity
                    best_id = speaker_id

            if best_id is None or best_sim < self.similarity_threshold:
                speaker_id, profile = self._create_profile(embedding)
                best_sim = 1.0
            else:
                profile = self._profiles[best_id]
                profile.update_embedding(embedding, self.update_alpha)
                speaker_id = best_id

            return speaker_id, profile, best_sim

    def register_segment(self, speaker_id: str, duration: float, similarity: float) -> None:
        with self._lock:
            profile = self._profiles.get(speaker_id)
            if profile is None:
                profile_id, profile = self._create_profile(np.zeros(1, dtype=np.float32))
                speaker_id = profile_id
            profile.statement_count += 1
            profile.duration += duration
            profile.last_similarity = similarity
            profile.updated_at = datetime.utcnow()

    def _assign_default(self) -> Tuple[str, SpeakerProfile, float]:
        with self._lock:
            if not self._profiles:
                speaker_id, profile = self._create_profile(np.zeros(1, dtype=np.float32))
            else:
                speaker_id = next(iter(self._profiles))
                profile = self._profiles[speaker_id]
            return speaker_id, profile, 0.0

    def _create_profile(self, embedding: np.ndarray) -> Tuple[str, SpeakerProfile]:
        speaker_id = f"speaker_{self._counter}"
        display_name = f"Speaker {self._counter}"
        normalized = embedding
        if np.any(embedding):
            normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name,
            embedding=normalized,
        )
        self._profiles[speaker_id] = profile
        self._counter = min(self._counter + 1, self.max_speakers + 1)
        return speaker_id, profile

    def get_profiles(self) -> List[SpeakerProfile]:
        with self._lock:
            return [profile for profile in self._profiles.values()]

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        with self._lock:
            return self._profiles.get(speaker_id)


class TimelineManager:
    def __init__(self) -> None:
        self._segments: List[TimelineSegment] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._segments = []

    def add_segment(self, speaker_id: str, speaker_name: str, start: datetime, end: datetime) -> None:
        with self._lock:
            if self._segments and self._segments[-1].speaker_id == speaker_id:
                self._segments[-1].end_time = end
            else:
                self._segments.append(
                    TimelineSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=start,
                        end_time=end,
                    )
                )

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [segment.to_dict() for segment in self._segments]

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._segments:
                return None
            return self._segments[-1].to_dict()


class TranscriptionService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.pipeline = None
        self.device = "cpu"
        self.torch_dtype: torch.dtype = torch.float32
        self.config: Optional[WhisperConfig] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.segment_callback: Optional[Callable[[TranscribedSegment], None]] = None
        self.lock = threading.Lock()

        # 동적 파라미터
        self.resample_up = 1
        self.resample_down = 1
        self.vad = SileroVAD()
        self.encoder = SpeechBrainEncoder()
        self.speakers = SpeakerHandler()
        self.timeline = TimelineManager()

    def initialize(self) -> None:
        """Whisper 파이프라인을 초기화한다."""

        with self.lock:
            if self.pipeline is not None:
                return

            raw_config = load_config(self.config_path)
            self.config = WhisperConfig(**raw_config)

            self.device, self.torch_dtype = select_device(self.config.force_device)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.device)
            processor = AutoProcessor.from_pretrained(self.config.model_id)

            pipe_kwargs = {
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "torch_dtype": self.torch_dtype,
                "device": self.device,
                "chunk_length_s": self.config.chunk_length_s,
                "stride_length_s": (
                    self.config.stride_seconds,
                    self.config.stride_seconds,
                ),
                "return_timestamps": True,
                "generate_kwargs": deep_update(
                    json.loads(json.dumps(DEFAULT_CONFIG["generate_kwargs"])),
                    self.config.generate_kwargs,
                ),
            }
            self.pipeline = pipeline("automatic-speech-recognition", **pipe_kwargs)

            factor = gcd(self.config.audio_source_sr, self.config.target_sr)
            self.resample_up = self.config.target_sr // factor
            self.resample_down = self.config.audio_source_sr // factor

            logger.info("Whisper 파이프라인 초기화 완료 (device=%s)", self.device)

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def reset_state(self) -> None:
        self.speakers.reset()
        self.timeline.reset()

    def start(self, callback: Callable[[TranscribedSegment], None]) -> None:
        """전사를 시작한다."""

        if self.pipeline is None:
            raise RuntimeError("Whisper 파이프라인이 초기화되지 않았습니다")

        if self.is_running():
            raise RuntimeError("이미 전사가 진행 중입니다")

        self.segment_callback = callback
        self.stop_event.clear()
        self.reset_state()
        self.audio_queue = queue.Queue(maxsize=self.config.queue_maxsize if self.config else 100)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("실시간 전사 스레드 시작")

    def stop(self) -> None:
        """전사를 중단한다."""

        self.stop_event.set()
        if self.audio_queue is not None:
            try:
                self.audio_queue.put_nowait(np.array([]))  # 깨우기
            except queue.Full:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.audio_queue = None
        logger.info("실시간 전사 스레드 종료")

    def speaker_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for profile in self.speakers.get_profiles():
            summaries.append(
                {
                    "speaker_id": profile.speaker_id,
                    "name": profile.display_name,
                    "count": profile.statement_count,
                    "duration": profile.duration,
                    "last_similarity": profile.last_similarity,
                    "updated_at": profile.updated_at.isoformat(),
                }
            )
        return summaries

    def timeline_snapshot(self) -> List[Dict[str, Any]]:
        return self.timeline.snapshot()

    def timeline_latest(self) -> Optional[Dict[str, Any]]:
        return self.timeline.latest()

    def _run(self) -> None:
        if self.config is None or self.pipeline is None:
            return

        blocksize = max(1, int(self.config.audio_source_sr * self.config.blocksize_seconds))
        batch_size = self.config.batch_size
        stride_seconds = self.config.stride_seconds
        chunk_duration = self.config.chunk_duration
        target_sr = self.config.target_sr

        full_audio_buffer: List[np.ndarray] = []
        prev_transcription = ""

        def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                logger.warning("SoundDevice 상태: %s", status)
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            if self.audio_queue is None:
                return
            audio_data = indata.copy().astype(np.float32)
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=self.config.audio_source_sr,
                blocksize=blocksize,
                channels=1,
                dtype="float32",
                callback=audio_callback,
            ):
                logger.info("마이크 입력을 통한 전사를 시작합니다")

                while not self.stop_event.is_set():
                    if self.audio_queue is None:
                        break
                    try:
                        chunk = self.audio_queue.get(timeout=chunk_duration)
                    except queue.Empty:
                        continue

                    if chunk.size == 0:
                        continue
                    if chunk.ndim > 1:
                        chunk = chunk[:, 0]

                    resampled_chunk = resample_poly(
                        chunk,
                        up=self.resample_up,
                        down=self.resample_down,
                    )
                    full_audio_buffer.append(resampled_chunk)

                    total_samples = sum(len(buf) for buf in full_audio_buffer)
                    total_duration = total_samples / target_sr
                    if total_duration < chunk_duration:
                        continue

                    audio_data_np = np.concatenate(full_audio_buffer)
                    if audio_data_np.size == 0:
                        full_audio_buffer = []
                        continue

                    if not self.vad.is_speech(audio_data_np, target_sr):
                        full_audio_buffer = []
                        continue

                    embedding = self.encoder.embed(audio_data_np, target_sr)
                    speaker_id, profile, similarity = self.speakers.classify(embedding)
                    segment_end = datetime.utcnow()
                    segment_start = segment_end - timedelta(seconds=total_duration)

                    with torch.inference_mode():
                        result = self.pipeline(
                            audio_data_np,
                            batch_size=batch_size,
                        )

                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    current_text = result.get("text", "").strip()
                    prev_text = prev_transcription.strip()
                    if current_text.startswith(prev_text) and len(current_text) > len(prev_text):
                        new_segment = current_text[len(prev_text):].strip()
                    else:
                        new_segment = current_text

                    if new_segment and self.segment_callback:
                        self.speakers.register_segment(speaker_id, total_duration, similarity)
                        self.timeline.add_segment(
                            speaker_id,
                            profile.display_name,
                            segment_start,
                            segment_end,
                        )
                        segment = TranscribedSegment(
                            text=new_segment,
                            speaker_id=speaker_id,
                            speaker_name=profile.display_name,
                            similarity=similarity,
                            start_time=segment_start,
                            end_time=segment_end,
                            duration=total_duration,
                        )
                        self.segment_callback(segment)

                    prev_transcription = current_text

                    overlap_samples = int(target_sr * stride_seconds)
                    if overlap_samples > 0 and audio_data_np.size > overlap_samples:
                        full_audio_buffer = [audio_data_np[-overlap_samples:]]
                    else:
                        full_audio_buffer = [audio_data_np]

        except Exception as exc:  # noqa: BLE001
            logger.exception("전사 중 오류 발생: %s", exc)
        finally:
            self.stop_event.set()


CONFIG_PATH = Path("config_whisper.json")
transcription_service = TranscriptionService(CONFIG_PATH)


@dataclass
class MeetingState:
    session_id: Optional[str] = None
    last_session_id: Optional[str] = None
    topic: str = ""
    speaker_id: str = "Speaker 1"
    is_active: bool = False


meeting_state = MeetingState()


@dataclass
class SpeakerStatsEntry:
    total_statements: int = 0
    topic_sum: float = 0.0
    novelty_sum: float = 0.0

    def add(self, topic_score: float, novelty_score: float) -> None:
        self.total_statements += 1
        self.topic_sum += topic_score
        self.novelty_sum += novelty_score

    def avg_topic(self) -> float:
        return self.topic_sum / self.total_statements if self.total_statements else 0.0

    def avg_novelty(self) -> float:
        return self.novelty_sum / self.total_statements if self.total_statements else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_statements": self.total_statements,
            "avg_topic_relevance": self.avg_topic(),
            "avg_novelty": self.avg_novelty(),
        }


@dataclass
class MeetingStatistics:
    total_statements: int = 0
    topic_sum: float = 0.0
    novelty_sum: float = 0.0
    speakers: Dict[str, SpeakerStatsEntry] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> None:
        with self.lock:
            self.total_statements = 0
            self.topic_sum = 0.0
            self.novelty_sum = 0.0
            self.speakers.clear()

    def add_statement(self, speaker_id: str, topic_score: float, novelty_score: float) -> None:
        with self.lock:
            self.total_statements += 1
            self.topic_sum += topic_score
            self.novelty_sum += novelty_score
            entry = self.speakers.setdefault(speaker_id, SpeakerStatsEntry())
            entry.add(topic_score, novelty_score)

    def overall_dict(self) -> Dict[str, Any]:
        with self.lock:
            avg_topic = self.topic_sum / self.total_statements if self.total_statements else 0.0
            avg_novelty = self.novelty_sum / self.total_statements if self.total_statements else 0.0
            return {
                "total_statements": self.total_statements,
                "avg_topic_relevance": avg_topic,
                "avg_novelty": avg_novelty,
            }

    def speaker_dict(self) -> Dict[str, Any]:
        with self.lock:
            data = {speaker: stats.to_dict() for speaker, stats in self.speakers.items()}
        for summary in transcription_service.speaker_summaries():
            speaker_id = summary["speaker_id"]
            entry = data.setdefault(
                speaker_id,
                {
                    "total_statements": summary["count"],
                    "avg_topic_relevance": 0.0,
                    "avg_novelty": 0.0,
                },
            )
            entry["total_duration"] = summary["duration"]
        return data


meeting_stats = MeetingStatistics()


meeting_history_lock = threading.Lock()
meeting_history: List[Dict[str, Any]] = []
last_report_path: Optional[Path] = None
last_summary_payload: Optional[Dict[str, Any]] = None

# 간단한 전사 중복 방지용 상태 (텍스트/화자/시간)
_last_transcription_text: Optional[str] = None
_last_transcription_speaker: Optional[str] = None
_last_transcription_ts: float = 0.0

REPORTS_DIR = Path("output") / "reports"


GEMMA_SYSTEM_PROMPT = (
    "당신은 전문적인 회의 평가 AI입니다.\n"
    "다음 기준으로 발언을 평가하세요:\n"
    "1. 주제일치성 (0-10): 회의 주제와의 관련성\n"
    "2. 신규성 (0-10): 새로운 정보나 관점 제공 정도\n"
    "3. 기여도: 회의 진행에 대한 기여 정도\n\n"
    "평가는 객관적이고 건설적이어야 하며, JSON 형식으로 응답하세요."
)


class OllamaEvaluator:
    """Ollama 서버의 Gemma 모델을 활용해 발언을 평가한다.

    변경된 모델 I/O에 맞춰 단일 프롬프트로 평가하며,
    화자별 누적 컨텍스트(speaker_context)를 유지한다.
    """

    def __init__(self, model: str = "gemma3-270m-local-e3", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.meeting_topic = ""
        self.lock = threading.Lock()
        # 화자별 이전 발언 누적 (현재 세션 한정)
        self._speaker_history: Dict[str, List[str]] = {}
        # 컨텍스트 최대 길이 (문자)
        self.max_ctx_chars: int = int(os.environ.get("SPEAKER_CONTEXT_CHARS", 400))

    def initialize(self, topic: str) -> None:
        with self.lock:
            self.meeting_topic = topic
            self._speaker_history.clear()

    def _clamp_tail(self, text: str, limit: int) -> str:
        if not text or limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return "…" + text[-(limit - 1) :]

    def _get_speaker_context_text(self, speaker_id: str) -> str:
        history = self._speaker_history.get(speaker_id) or []
        context_text = " ".join(history).strip()
        return self._clamp_tail(context_text, self.max_ctx_chars)

    def _append_speaker_sentence(self, speaker_id: str, sentence: str) -> None:
        if not sentence:
            return
        history = self._speaker_history.setdefault(speaker_id, [])
        history.append(sentence)
        # 간단한 메모리 제어: 너무 많은 항목이면 앞부분 제거
        if len(history) > 200:
            del history[: len(history) - 200]

    def _build_prompt(self, topic: str, speaker_id: str, speaker_context: str, sentence: str) -> str:
        # 파인튜닝 데이터와 동일한 형식
        return (
            f"topic: {topic}\n"
            f"speaker_id: {speaker_id}\n"
            f"speaker_context: {speaker_context or '없음'}\n"
            f"sentence: {sentence}\n"
        )

    def _call_model(self, prompt: str, timeout: float = 60.0) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            logger.error("Ollama 요청 실패: %s", exc)
            return ""
        except ValueError as exc:
            logger.error("Ollama 응답 파싱 실패: %s", exc)
            return ""

    def _parse_reason_and_scores(self, response_text: str) -> Tuple[str, float, float]:
        text = (response_text or "").strip()
        if not text:
            return "", 5.0, 5.0

        # 괄호로 끝나는 "… (x, y)" 패턴 우선 추출
        m = re.search(r"\(([^)]*)\)\s*$", text)
        topic_score = 5.0
        novelty_score = 5.0
        reason = text
        if m:
            inside = m.group(1)
            # 쉼표 또는 공백 구분 허용
            parts = [p.strip() for p in re.split(r"[,\s]+", inside) if p.strip()]
            if len(parts) >= 2:
                try:
                    topic_score = float(parts[0])
                    novelty_score = float(parts[1])
                except ValueError:
                    pass
            # 괄호 제거한 나머지를 이유로 사용
            reason = text[: m.start()].rstrip()

        # 0~10 범위로 클램프
        topic_score = max(0.0, min(10.0, topic_score))
        novelty_score = max(0.0, min(10.0, novelty_score))
        return reason, topic_score, novelty_score

    def evaluate(self, sentence: str, speaker_id: str) -> Dict[str, Any]:
        """단일 프롬프트(주제/화자/컨텍스트/문장)로 평가 후 점수/코멘트 반환.

        반환: {"topic_relevance": float, "novelty": float, "comment": str}
        """
        # 현재 화자의 이전 발언으로 컨텍스트 구성
        with self.lock:
            topic = self.meeting_topic
            speaker_context = self._get_speaker_context_text(speaker_id)

        prompt = self._build_prompt(topic, speaker_id, speaker_context, sentence)
        raw = self._call_model(prompt)
        reason, topic_score, novelty_score = self._parse_reason_and_scores(raw)

        # 평가 이후 현재 문장을 화자 컨텍스트에 추가
        with self.lock:
            self._append_speaker_sentence(speaker_id, sentence)

        return {
            "topic_relevance": topic_score,
            "novelty": novelty_score,
            "comment": reason,
        }

OLLAMA_MODEL = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma3-270m-local-e3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

ollama_evaluator = OllamaEvaluator(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


async def _broadcast_worker() -> None:
    """전달 대기열에서 메시지를 받아 WebSocket으로 브로드캐스트한다."""

    queue_: asyncio.Queue = app.state.event_queue
    while True:
        message = await queue_.get()
        try:
            await manager.broadcast(message)
        finally:
            queue_.task_done()


def enqueue_event(message: Dict[str, Any]) -> None:
    """비동기 브로드캐스트를 위해 메시지를 대기열에 넣는다."""

    loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "event_loop", None)
    queue_: Optional[asyncio.Queue] = getattr(app.state, "event_queue", None)
    if loop is None or queue_ is None:
        logger.warning("이벤트 루프 또는 큐가 준비되지 않았습니다")
        return
    asyncio.run_coroutine_threadsafe(queue_.put(message), loop)


def broadcast_session_status() -> None:
    enqueue_event(
        {
            "type": "session_status",
            "is_active": meeting_state.is_active,
            "session_id": meeting_state.session_id,
            "topic": meeting_state.topic,
        }
    )


def build_speaker_stats_payload(
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    speaker_summaries = transcription_service.speaker_summaries()
    total_duration = sum(item["duration"] for item in speaker_summaries) or 0.0
    speakers_payload: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for summary in speaker_summaries:
        speaker_id = summary["speaker_id"]
        stats = speaker_stats.get(speaker_id, {})
        duration = summary["duration"]
        participation = (duration / total_duration * 100.0) if total_duration else 0.0
        speakers_payload.append(
            {
                "speaker_id": speaker_id,
                "name": summary["name"],
                "count": summary["count"],
                "duration": duration,
                "topic_avg": stats.get("avg_topic_relevance", 0.0) or 0.0,
                "novelty_avg": stats.get("avg_novelty", 0.0) or 0.0,
                "participation": participation,
            }
        )
        seen.add(speaker_id)

    for speaker_id, stats in speaker_stats.items():
        if speaker_id in seen:
            continue
        speakers_payload.append(
            {
                "speaker_id": speaker_id,
                "name": speaker_id,
                "count": stats.get("total_statements", 0) or 0,
                "duration": 0.0,
                "topic_avg": stats.get("avg_topic_relevance", 0.0) or 0.0,
                "novelty_avg": stats.get("avg_novelty", 0.0) or 0.0,
                "participation": 0.0,
            }
        )

    speakers_payload.sort(key=lambda item: item["speaker_id"])
    return speakers_payload


def broadcast_stats_messages() -> None:
    overall_stats = meeting_stats.overall_dict()
    speaker_stats = meeting_stats.speaker_dict()

    enqueue_event(
        {
            "type": "stats_update",
            "overall_stats": overall_stats,
            "speaker_stats": speaker_stats,
        }
    )

    enqueue_event(
        {
            "type": "stats",
            "avg_topic": overall_stats.get("avg_topic_relevance", 0.0) or 0.0,
            "avg_novelty": overall_stats.get("avg_novelty", 0.0) or 0.0,
            "speakers": build_speaker_stats_payload(overall_stats, speaker_stats),
        }
    )


def handle_transcription(segment: TranscribedSegment) -> None:
    cleaned = segment.text.strip()
    if not cleaned or not meeting_state.is_active:
        return
    meeting_state.speaker_id = segment.speaker_id
    # Deduplicate bursts due to overlapping stride/ASR repeat
    global _last_transcription_text, _last_transcription_speaker, _last_transcription_ts
    now_ts = time.time()
    if (
        _last_transcription_text == cleaned
        and _last_transcription_speaker == segment.speaker_id
        and (now_ts - _last_transcription_ts) < 1.5
    ):
        return
    _last_transcription_text = cleaned
    _last_transcription_speaker = segment.speaker_id
    _last_transcription_ts = now_ts
    logger.debug("새 전사: %s", cleaned)
    enqueue_event(
        {
            "type": "transcription",
            "text": cleaned,
            "speaker_id": segment.speaker_id,
            "speaker_name": segment.speaker_name,
            "similarity": segment.similarity,
            "duration": segment.duration,
            "timestamp": segment.end_time.isoformat(),
        }
    )
    schedule_evaluation(segment)
    latest_segment = transcription_service.timeline_latest()
    if latest_segment:
        enqueue_event(
            {
                "type": "timeline_segment",
                "segment": latest_segment,
            }
        )


def schedule_evaluation(segment: TranscribedSegment) -> None:
    loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "event_loop", None)
    queue_: Optional[asyncio.Queue] = getattr(app.state, "evaluation_queue", None)
    if loop is None or queue_ is None:
        logger.warning("평가 큐가 준비되지 않았습니다")
        return
    payload = {
        "text": segment.text.strip(),
        "speaker_id": segment.speaker_id,
        "speaker_name": segment.speaker_name,
        "duration": segment.duration,
        "timestamp": segment.end_time.isoformat(),
        "session_id": meeting_state.session_id,
    }
    asyncio.run_coroutine_threadsafe(queue_.put(payload), loop)


async def _evaluation_worker() -> None:
    queue_: asyncio.Queue = app.state.evaluation_queue
    loop = asyncio.get_running_loop()
    while True:
        task = await queue_.get()
        try:
            await loop.run_in_executor(
                None,
                lambda t=task: _evaluate_and_broadcast(t),
            )
        finally:
            queue_.task_done()


def _evaluate_and_broadcast(task: Dict[str, Any]) -> None:
    text = task.get("text", "")
    speaker_id = task.get("speaker_id", "Speaker 1")
    session_id = task.get("session_id")
    speaker_name = task.get("speaker_name", speaker_id)
    duration = float(task.get("duration", 0.0) or 0.0)
    timestamp_str = task.get("timestamp")
    if not text or session_id is None:
        return
    if not meeting_state.is_active or meeting_state.session_id != session_id:
        return
    try:
        result = ollama_evaluator.evaluate(text, speaker_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("평가 실패: %s", exc)
        result = {"topic_relevance": 5.0, "novelty": 5.0, "comment": ""}

    meeting_stats.add_statement(speaker_id, result["topic_relevance"], result["novelty"])
    with meeting_history_lock:
        meeting_history.append(
            {
                "text": text,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "duration": duration,
                "topic_relevance": result["topic_relevance"],
                "novelty": result["novelty"],
                "timestamp": timestamp_str or datetime.now().isoformat(),
                "comment": result.get("comment", ""),
            }
        )

    enqueue_event(
        {
            "type": "evaluation",
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "text": text,
            "topic_relevance": result["topic_relevance"],
            "novelty": result["novelty"],
            "timestamp": timestamp_str or datetime.now().isoformat(),
            "comment": result.get("comment", ""),
        }
    )

    broadcast_stats_messages()


def _extract_json_object(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = response_text[start : end + 1]
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def create_summary_with_gemma(
    topic: str,
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not history:
        return {
            "title": f"{topic} 회의 요약",
            "overview": "회의 발언 데이터가 없어 요약을 생성하지 못했습니다.",
            "key_points": [],
            "insights": [],
            "speaker_analysis": [],
            "metrics": overall_stats,
            "recommendations": [],
            "conclusion": "",
        }

    recent_history = history[-40:]
    history_lines = [
        (
            f"- {item['timestamp']} | {item['speaker_id']}: {item['text']}"
            f" (주제 일치 {item['topic_relevance']:.1f}, 신규성 {item['novelty']:.1f})"
        )
        for item in recent_history
    ]

    prompt = (
        f"{GEMMA_SYSTEM_PROMPT}\n\n"
        "다음 회의 기록을 바탕으로 전문적인 분석 보고서를 작성하세요."
        "보고서는 JSON 형식으로 응답하며, 키는 다음과 같습니다:\n"
        "{\n"
        "  \"title\": 회의 제목 문자열,\n"
        "  \"overview\": 회의 전반 요약 (문단),\n"
        "  \"key_points\": 주요 논의 사항 목록,\n"
        "  \"insights\": 핵심 인사이트 목록,\n"
        "  \"speaker_analysis\": [ {\"speaker\", \"summary\", \"topic_score\", \"novelty_score\", \"contribution\"} ],\n"
        "  \"metrics\": {\"avg_topic\", \"avg_novelty\", \"total_statements\", \"participation_level\"},\n"
        "  \"recommendations\": 개선 제안 목록,\n"
        "  \"conclusion\": 결론 요약\n"
        "}\n"
        "JSON 이외의 텍스트는 포함하지 마세요.\n\n"
        f"회의 주제: {topic}\n"
        f"전체 통계: {json.dumps(overall_stats, ensure_ascii=False)}\n"
        f"화자별 통계: {json.dumps(speaker_stats, ensure_ascii=False)}\n"
        "최근 발언 로그:\n"
        f"{os.linesep.join(history_lines)}\n\n"
        "보고서를 생성하세요."
    )

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
        parsed = _extract_json_object(raw)
        if parsed:
            return parsed
        logger.warning("Gemma 응답에서 JSON을 추출하지 못했습니다. 원문: %s", raw[:200])
    except requests.exceptions.RequestException as exc:
        logger.error("Gemma 요약 생성 실패: %s", exc)

    return {
        "title": f"{topic} 회의 요약",
        "overview": "Gemma 요약 생성에 실패하여 기본 통계만 제공합니다.",
        "key_points": [],
        "insights": [],
        "speaker_analysis": [
            {
                "speaker": speaker,
                "summary": "요약 생성 실패",
                "topic_score": stats.get("avg_topic_relevance", 0.0),
                "novelty_score": stats.get("avg_novelty", 0.0),
                "contribution": stats.get("total_statements", 0),
            }
            for speaker, stats in speaker_stats.items()
        ],
        "metrics": {
            "avg_topic": overall_stats.get("avg_topic_relevance", 0.0),
            "avg_novelty": overall_stats.get("avg_novelty", 0.0),
            "total_statements": overall_stats.get("total_statements", 0),
            "participation_level": "",
        },
        "recommendations": [],
        "conclusion": "",
    }


def render_pdf_report(
    summary: Dict[str, Any],
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab 라이브러리가 설치되어 있지 않습니다. 'pip install reportlab' 후 다시 시도하세요.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=1,
        fontSize=18,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        textColor="white",
        backColor="#1F4E79",
        alignment=0,
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        leftIndent=4,
        rightIndent=4,
        leading=14,
    )
    text_style = styles["BodyText"]
    text_style.leading = 14

    story: List[Any] = []
    story.append(Paragraph(summary.get("title", "회의 보고서"), title_style))
    story.append(Paragraph("회의 개요", section_style))
    story.append(Paragraph(summary.get("overview", ""), text_style))

    if summary.get("key_points"):
        story.append(Paragraph("주요 논의 사항", section_style))
        for idx, point in enumerate(summary["key_points"], start=1):
            story.append(Paragraph(f"{idx}. {point}", text_style))
            story.append(Spacer(1, 4))

    if summary.get("insights"):
        story.append(Paragraph("핵심 인사이트", section_style))
        for idx, insight in enumerate(summary["insights"], start=1):
            story.append(Paragraph(f"{idx}. {insight}", text_style))
            story.append(Spacer(1, 4))

    speaker_data = summary.get("speaker_analysis") or []
    if speaker_data:
        story.append(Paragraph("발언자별 평가", section_style))
        table_data = [["화자", "요약", "주제 점수", "신규성", "기여"]]
        for item in speaker_data:
            table_data.append(
                [
                    item.get("speaker", "-"),
                    item.get("summary", ""),
                    str(item.get("topic_score", "")),
                    str(item.get("novelty_score", "")),
                    str(item.get("contribution", "")),
                ]
            )

        table = Table(table_data, colWidths=[25 * mm, 65 * mm, 20 * mm, 20 * mm, 20 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E1F2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.gray),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.gray),
                ]
            )
        )
        story.append(table)

    metrics = summary.get("metrics", {})
    if metrics:
        story.append(Paragraph("회의 통계", section_style))
        metrics_table = Table(
            [
                ["평균 주제일치성", str(metrics.get("avg_topic", "")), "평균 신규성", str(metrics.get("avg_novelty", ""))],
                ["총 발언 수", str(metrics.get("total_statements", "")), "참여도", str(metrics.get("participation_level", ""))],
            ],
            colWidths=[35 * mm, 35 * mm, 35 * mm, 35 * mm],
        )
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E9EDF5")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F4E79")),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.gray),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.gray),
                ]
            )
        )
        story.append(metrics_table)

    if summary.get("recommendations"):
        story.append(Paragraph("결론 및 제안", section_style))
        for idx, rec in enumerate(summary["recommendations"], start=1):
            story.append(Paragraph(f"{idx}. {rec}", text_style))
            story.append(Spacer(1, 4))

    if summary.get("conclusion"):
        story.append(Paragraph("최종 결론", section_style))
        story.append(Paragraph(summary["conclusion"], text_style))

    if history:
        story.append(Paragraph("부록 - 발언 로그", section_style))
        for item in history[-20:]:
            story.append(
                Paragraph(
                    f"{item['timestamp']} | {item['speaker_id']} : {item['text']} (주제 {item['topic_relevance']:.1f}, 신규성 {item['novelty']:.1f})",
                    text_style,
                )
            )

    doc.build(story)


def finalize_meeting_report(
    session_id: Optional[str],
    topic: str,
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> None:
    global last_report_path, last_summary_payload

    if session_id is None:
        logger.warning("세션 ID 없이 보고서를 생성할 수 없습니다")
        return

    summary = create_summary_with_gemma(topic, overall_stats, speaker_stats, history)
    last_summary_payload = summary

    try:
        report_path = REPORTS_DIR / f"meeting_report_{session_id}.pdf"
        render_pdf_report(summary, overall_stats, speaker_stats, history, report_path)
        last_report_path = report_path
        logger.info("PDF 보고서 생성 완료: %s", report_path)
        enqueue_event(
            {
                "type": "report_ready",
                "available": True,
                "path": str(report_path),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF 보고서 생성 실패: %s", exc)
        last_report_path = None
        enqueue_event(
            {
                "type": "report_ready",
                "available": False,
                "detail": str(exc),
            }
        )


def get_dashboard_html() -> str:
    """MeetingProgram의 대시보드 HTML을 반환한다."""
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MeetingProgram - 실시간 회의 평가</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-4">
        <header class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">🎯 MeetingProgram</h1>
            <p class="text-gray-600">실시간 회의 평가 시스템 (Whisper + Gemma 1B)</p>
        </header>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">회의 제어</h2>
            <div class="flex flex-col md:flex-row md:space-x-4 mb-4 space-y-2 md:space-y-0">
                <input type="text" id="meetingTopic" placeholder="회의 주제" class="flex-1 p-2 border rounded" required>
                <select id="speakerId" class="w-48 p-2 border rounded">
                    <option value="Speaker 1">Speaker 1</option>
                </select>
            </div>
            <div class="flex flex-wrap gap-4">
                <button id="startBtn" class="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded">
                    🎤 회의 시작
                </button>
                <button id="stopBtn" class="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded" disabled>
                    ⏹️ 회의 중지
                </button>
                <button id="reportBtn" class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded" disabled>
                    📄 보고서 다운로드
                </button>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold mb-4">현재 상태</h3>
                <div id="status" class="text-gray-600">대기 중...</div>
                <div id="currentSpeaker" class="mt-2 text-sm text-gray-500"></div>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold mb-4">전체 통계</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="text-sm text-gray-500">총 발언</div>
                        <div id="totalStatements" class="text-2xl font-bold">0</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-500">평균 주제일치성</div>
                        <div id="avgTopic" class="text-2xl font-bold">0.0</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-500">평균 신규성</div>
                        <div id="avgNovelty" class="text-2xl font-bold">0.0</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h3 class="text-lg font-semibold mb-4">실시간 발언</h3>
            <div id="transcriptionFeed" class="space-y-2 max-h-96 overflow-y-auto">
                <div class="text-gray-500 text-center">발언을 기다리는 중...</div>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold mb-4">화자별 통계</h3>
            <div id="speakerStats" class="overflow-x-auto">
                <table class="min-w-full table-auto">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-2 text-left">화자</th>
                            <th class="px-4 py-2 text-left">발언 수</th>
                            <th class="px-4 py-2 text-left">주제일치성</th>
                            <th class="px-4 py-2 text-left">신규성</th>
                        </tr>
                    </thead>
                    <tbody id="speakerStatsBody">
                        <tr>
                            <td colspan="4" class="px-4 py-2 text-center text-gray-500">데이터 없음</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        let isConnected = false;
        let currentSession = null;

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const reportBtn = document.getElementById('reportBtn');

        ws.onopen = () => { isConnected = true; };
        ws.onclose = () => { isConnected = false; };
        ws.onmessage = (event) => handleWebSocketMessage(JSON.parse(event.data));

        function handleWebSocketMessage(data) {
            switch (data.type) {
                case 'session_status':
                    updateSessionStatus(data);
                    break;
                case 'transcription':
                    addTranscription(data);
                    break;
                case 'evaluation':
                    appendEvaluation(data);
                    break;
                case 'stats_update':
                    updateStats(data);
                    break;
                case 'report_ready':
                    updateReportStatus(data);
                    break;
            }
        }

        function updateSessionStatus(data) {
            const status = document.getElementById('status');
            if (data.is_active) {
                status.textContent = `🎤 회의 진행 중 (주제: ${data.topic})`;
                status.className = 'text-green-600 font-semibold';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                reportBtn.disabled = true;
                currentSession = data.session_id;
            } else {
                status.textContent = '⏸️ 대기 중';
                status.className = 'text-gray-600';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                reportBtn.disabled = true;
                currentSession = null;
            }
        }

        function updateReportStatus(data) {
            if (data.available) {
                reportBtn.disabled = false;
                reportBtn.textContent = '📄 보고서 다운로드';
            } else {
                reportBtn.disabled = true;
                reportBtn.textContent = '📄 보고서 준비 중...';
            }
        }

        function addTranscription(data) {
            const feed = document.getElementById('transcriptionFeed');
            const item = document.createElement('div');
            item.className = 'border-l-4 border-blue-400 pl-4 py-2';
            item.innerHTML = `
                <div class="flex justify-between">
                    <span class="font-semibold text-blue-600">${data.speaker_id}</span>
                    <span class="text-sm text-gray-500">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="mt-1">${data.text}</div>
            `;

            const firstElement = feed.firstElementChild;
            if (firstElement && firstElement.classList.contains('text-gray-500')) {
                feed.removeChild(firstElement);
            }
            feed.prepend(item);
            while (feed.children.length > 20) {
                feed.removeChild(feed.lastElementChild);
            }
        }

        function appendEvaluation(data) {
            const feed = document.getElementById('transcriptionFeed');
            const firstElement = feed.firstElementChild;
            if (firstElement && !firstElement.classList.contains('text-gray-500')) {
                const badge = document.createElement('div');
                badge.className = 'mt-2 text-sm';
                badge.innerHTML = `
                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded">주제: ${data.topic_relevance.toFixed(1)}</span>
                    <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded ml-2">신규성: ${data.novelty.toFixed(1)}</span>
                `;
                firstElement.appendChild(badge);
            }
        }

        function updateStats(data) {
            const stats = data.overall_stats;
            document.getElementById('totalStatements').textContent = stats.total_statements || 0;
            document.getElementById('avgTopic').textContent = (stats.avg_topic_relevance || 0).toFixed(1);
            document.getElementById('avgNovelty').textContent = (stats.avg_novelty || 0).toFixed(1);

            const tbody = document.getElementById('speakerStatsBody');
            tbody.innerHTML = '';
            if (!data.speaker_stats || Object.keys(data.speaker_stats).length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="px-4 py-2 text-center text-gray-500">데이터 없음</td></tr>';
                return;
            }
            Object.entries(data.speaker_stats).forEach(([speaker, stats]) => {
                const row = document.createElement('tr');
                row.className = 'border-t';
                row.innerHTML = `
                    <td class="px-4 py-2 font-semibold">${speaker}</td>
                    <td class="px-4 py-2">${stats.total_statements}</td>
                    <td class="px-4 py-2">${stats.avg_topic_relevance.toFixed(1)}</td>
                    <td class="px-4 py-2">${stats.avg_novelty.toFixed(1)}</td>
                `;
                tbody.appendChild(row);
            });
        }

        startBtn.addEventListener('click', async () => {
            const topic = document.getElementById('meetingTopic').value.trim();
            const speakerId = document.getElementById('speakerId').value;
            if (!topic) {
                alert('회의 주제를 입력하세요.');
                return;
            }
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic, speaker_id: speakerId })
            });
            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || '회의 시작 실패');
            }
        });

        stopBtn.addEventListener('click', async () => {
            if (!currentSession) return;
            const response = await fetch(`/api/stop/${currentSession}`, { method: 'POST' });
            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || '회의 중지 실패');
            }
        });

        reportBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/report');
                if (!response.ok) {
                    const error = await response.json();
                    alert(error.detail || '보고서를 가져올 수 없습니다.');
                    return;
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'meeting_report.pdf';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } catch (error) {
                alert('오류: ' + error.message);
            }
        });
    </script>
</body>
</html>
    """


@app.on_event("startup")
async def on_startup() -> None:
    app.state.event_loop = asyncio.get_running_loop()
    app.state.event_queue = asyncio.Queue()
    app.state.broadcast_task = asyncio.create_task(_broadcast_worker())
    app.state.evaluation_queue = asyncio.Queue()
    app.state.evaluation_task = asyncio.create_task(_evaluation_worker())
    logger.info("Background workers started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    for attr in ("broadcast_task", "evaluation_task"):
        task: Optional[asyncio.Task] = getattr(app.state, attr, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
@app.get("/", response_class=HTMLResponse)
async def get_main_page() -> HTMLResponse:
    return HTMLResponse(content=get_dashboard_html())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/speakers")
async def get_speakers() -> Dict[str, Any]:
    return {
        "is_active": meeting_state.is_active,
        "speakers": transcription_service.speaker_summaries(),
        "stats": meeting_stats.speaker_dict(),
        "stats_overall": meeting_stats.overall_dict(),
    }


@app.get("/api/timeline")
async def get_timeline() -> Dict[str, Any]:
    return {
        "is_active": meeting_state.is_active,
        "segments": transcription_service.timeline_snapshot(),
    }


@app.post("/api/start")
async def start_meeting(payload: Dict[str, Any]) -> Dict[str, Any]:
    if meeting_state.is_active:
        raise HTTPException(status_code=400, detail="이미 회의가 진행 중입니다")

    topic = (payload.get("topic") or "").strip()
    speaker_id = (payload.get("speaker_id") or "Speaker 1").strip() or "Speaker 1"
    if not topic:
        raise HTTPException(status_code=400, detail="회의 주제를 입력하세요")

    session_id = str(uuid.uuid4())[:8]
    meeting_state.session_id = session_id
    meeting_state.last_session_id = session_id
    meeting_state.topic = topic
    meeting_state.speaker_id = speaker_id

    meeting_stats.reset()
    ollama_evaluator.initialize(topic)
    global last_report_path, last_summary_payload
    with meeting_history_lock:
        meeting_history.clear()
    last_report_path = None
    last_summary_payload = None

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, transcription_service.initialize)
        await loop.run_in_executor(None, lambda: transcription_service.start(handle_transcription))
    except Exception as exc:  # noqa: BLE001
        meeting_state.session_id = None
        meeting_state.last_session_id = None
        meeting_state.topic = ""
        meeting_state.speaker_id = "Speaker 1"
        logger.exception("Whisper 전사 시작 실패: %s", exc)
        raise HTTPException(status_code=500, detail="전사 시작에 실패했습니다. 로그를 확인하세요.")

    meeting_state.is_active = True
    broadcast_session_status()
    broadcast_stats_messages()
    logger.info("회의 세션 시작: %s (주제=%s)", session_id, topic)

    return {"success": True, "session_id": session_id, "topic": topic}


@app.post("/api/stop/{session_id}")
async def stop_meeting(session_id: str) -> Dict[str, Any]:
    if not meeting_state.is_active:
        if session_id == meeting_state.last_session_id:
            return {"success": True, "message": "회의가 이미 종료되었습니다."}
        raise HTTPException(status_code=400, detail="진행 중인 세션이 없습니다")

    if meeting_state.session_id != session_id:
        if session_id == meeting_state.last_session_id:
            return {"success": True, "message": "회의가 이미 종료되었습니다."}
        raise HTTPException(status_code=400, detail="세션 ID가 일치하지 않습니다")

    loop = asyncio.get_running_loop()
    session_id_current = meeting_state.session_id
    topic = meeting_state.topic
    overall_snapshot = meeting_stats.overall_dict()
    speaker_snapshot = meeting_stats.speaker_dict()
    with meeting_history_lock:
        history_snapshot = list(meeting_history)

    try:
        await loop.run_in_executor(None, transcription_service.stop)
    except Exception as exc:  # noqa: BLE001
        logger.exception("전사 중지 실패: %s", exc)
        raise HTTPException(status_code=500, detail="전사 중지에 실패했습니다")

    meeting_state.is_active = False
    broadcast_session_status()

    await loop.run_in_executor(
        None,
        lambda: finalize_meeting_report(
            session_id_current,
            topic,
            overall_snapshot,
            speaker_snapshot,
            history_snapshot,
        ),
    )

    broadcast_stats_messages()
    logger.info("회의 세션 종료: %s", session_id)

    meeting_state.last_session_id = session_id_current
    meeting_state.session_id = None
    meeting_state.topic = ""
    meeting_state.speaker_id = "Speaker 1"

    return {"success": True, "message": "회의가 중지되었습니다."}


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    return {
        "is_active": meeting_state.is_active,
        "session_id": meeting_state.session_id,
        "topic": meeting_state.topic,
        "speaker_id": meeting_state.speaker_id,
    }


@app.get("/api/report")
async def get_report(format: str = "pdf") -> Any:  # noqa: ANN401 - FastAPI response
    if format == "json":
        if last_summary_payload:
            return JSONResponse(content=last_summary_payload)
        raise HTTPException(status_code=404, detail="요약이 아직 생성되지 않았습니다")

    if last_report_path and last_report_path.exists():
        return FileResponse(
            path=last_report_path,
            media_type="application/pdf",
            filename=last_report_path.name,
        )

    if last_summary_payload:
        return JSONResponse(
            status_code=202,
            content={
                "detail": "PDF가 아직 준비되지 않았습니다. JSON 요약을 format=json 으로 요청하세요.",
            },
        )

    raise HTTPException(status_code=404, detail="보고서가 아직 생성되지 않았습니다")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("whisper_web_ui:app", host="0.0.0.0", port=8000, reload=False)
