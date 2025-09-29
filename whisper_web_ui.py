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
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import requests
import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from diarization_service import MAX_SPEAKERS, DiarizationSegment, DiarizationService

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
    "chunk_length_s": 12,
    "stride_seconds": 0.4,
    "chunk_duration": 5.0,
    "blocksize_seconds": 0.2,
    "queue_maxsize": 100,
    "batch_size": 1,
    "silence_rms_threshold": 0.005,
    "generate_kwargs": {
        "task": "transcribe",
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "repetition_penalty": 1.05,
    },
}


class RollingWaveform:
    """오디오 파형을 일정 구간 유지하며 축약한다."""

    def __init__(
        self,
        points_per_second: int = 20,
        window_seconds: int = 120,
    ) -> None:
        self.points_per_second = points_per_second
        self.window_seconds = window_seconds
        self.max_points = max(1, points_per_second * window_seconds)
        self.points: deque[float] = deque(maxlen=self.max_points)
        self.samples_per_point: Optional[int] = None
        self._remainder = np.zeros(0, dtype=np.float32)

    def append_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size == 0:
            return
        if self.samples_per_point is None:
            self.samples_per_point = max(1, int(sample_rate / self.points_per_second))
        if self.samples_per_point <= 0:
            self.samples_per_point = 1

        if self._remainder.size:
            audio = np.concatenate((self._remainder, audio))

        full_points = audio.size // self.samples_per_point
        if full_points:
            trimmed = audio[: full_points * self.samples_per_point]
            segments = trimmed.reshape(full_points, self.samples_per_point)
            values = np.max(np.abs(segments), axis=1)
            for value in values:
                self.points.append(float(value))
        self._remainder = audio[full_points * self.samples_per_point :]

    def get_points(self) -> List[float]:
        if not self.points:
            return []
        max_val = max(self.points)
        if max_val <= 0.0:
            return [0.0 for _ in self.points]
        return [value / max_val for value in self.points]

    def window_duration(self) -> float:
        if not self.points:
            return 0.0
        return min(len(self.points), self.max_points) / self.points_per_second

    def reset(self) -> None:
        self.points.clear()
        self.samples_per_point = None
        self._remainder = np.zeros(0, dtype=np.float32)


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
    """Whisper 전사에 필요한 설정"""

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


class TranscriptionService:
    """Whisper 실시간 전사를 관리한다."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.pipeline = None
        self.device = "cpu"
        self.torch_dtype: torch.dtype = torch.float32
        self.config: Optional[WhisperConfig] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.segment_callback: Optional[Callable[[str], None]] = None
        self.lock = threading.Lock()

        # 동적 파라미터
        self.resample_up = 1
        self.resample_down = 1
        self.diarization_service: Optional[DiarizationService] = None
        self.waveform_buffer = RollingWaveform()
        self.last_waveform_emit = 0.0
        self.waveform_emit_interval = 0.5

    def attach_diarization(self, service: Optional[DiarizationService]) -> None:
        """다이어리제이션 서비스를 연결한다."""

        self.diarization_service = service

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

    def start(self, callback: Callable[[str], None]) -> None:
        """전사를 시작한다."""

        if self.pipeline is None:
            raise RuntimeError("Whisper 파이프라인이 초기화되지 않았습니다")

        if self.is_running():
            raise RuntimeError("이미 전사가 진행 중입니다")

        self.segment_callback = callback
        self.stop_event.clear()
        self.audio_queue = queue.Queue(maxsize=self.config.queue_maxsize if self.config else 100)
        self.waveform_buffer.reset()
        self.last_waveform_emit = 0.0
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
        self.waveform_buffer.reset()
        self.last_waveform_emit = 0.0
        logger.info("실시간 전사 스레드 종료")

    def _run(self) -> None:
        if self.config is None or self.pipeline is None:
            return

        blocksize = max(1, int(self.config.audio_source_sr * self.config.blocksize_seconds))
        batch_size = self.config.batch_size
        silence_rms_threshold = self.config.silence_rms_threshold
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
                    if self.diarization_service:
                        self.diarization_service.add_audio(resampled_chunk, target_sr)
                    self.waveform_buffer.append_chunk(resampled_chunk, target_sr)
                    now = time.time()
                    if now - self.last_waveform_emit >= self.waveform_emit_interval:
                        points = self.waveform_buffer.get_points()
                        if points:
                            enqueue_event(
                                {
                                    "type": "waveform",
                                    "points": points,
                                    "window_seconds": self.waveform_buffer.window_duration(),
                                }
                            )
                        self.last_waveform_emit = now
                    full_audio_buffer.append(resampled_chunk)

                    total_samples = sum(len(buf) for buf in full_audio_buffer)
                    total_duration = total_samples / target_sr
                    if total_duration < chunk_duration:
                        continue

                    audio_data_np = np.concatenate(full_audio_buffer)
                    rms_level = float(np.sqrt(np.mean(np.square(audio_data_np)))) if audio_data_np.size else 0.0
                    if rms_level < silence_rms_threshold:
                        full_audio_buffer = []
                        continue

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
                        self.segment_callback(new_segment)

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
diarization_service: Optional[DiarizationService] = None


@dataclass
class MeetingState:
    session_id: Optional[str] = None
    topic: str = ""
    speaker_id: str = "Speaker 1"
    is_active: bool = False
    expected_speakers: int = 2


meeting_state = MeetingState()


meeting_state_lock = threading.Lock()


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
            return {speaker: stats.to_dict() for speaker, stats in self.speakers.items()}


meeting_stats = MeetingStatistics()


meeting_history_lock = threading.Lock()
meeting_history: List[Dict[str, Any]] = []
last_report_path: Optional[Path] = None
last_summary_payload: Optional[Dict[str, Any]] = None

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
    """Ollama 서버의 Gemma 모델을 활용해 발언을 평가한다."""

    def __init__(self, model: str = "gemma3:1B", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.meeting_topic = ""
        self.history: List[str] = []
        self.lock = threading.Lock()

    def initialize(self, topic: str) -> None:
        with self.lock:
            self.meeting_topic = topic
            self.history.clear()

    def evaluate(self, text: str, speaker_id: str) -> Dict[str, float]:
        with self.lock:
            recent_texts = list(self.history[-3:])

        topic_prompt = self._create_topic_prompt(text)
        novelty_prompt = self._create_novelty_prompt(text, recent_texts)

        topic_score = self._query_model(topic_prompt, "topic_relevance", default=5.0)
        novelty_default = 8.0 if not recent_texts else 5.0
        novelty_score = self._query_model(novelty_prompt, "novelty", default=novelty_default)

        with self.lock:
            self.history.append(text)

        return {
            "topic_relevance": topic_score,
            "novelty": novelty_score,
        }

    def _create_topic_prompt(self, text: str) -> str:
        return (
            f"{GEMMA_SYSTEM_PROMPT}\n\n"
            f"회의 주제: {self.meeting_topic}\n"
            f'평가할 발언: "{text}"\n\n'
            "이 발언이 회의 주제와 얼마나 관련이 있는지 0-10점으로 평가하세요.\n"
            "0점: 전혀 관련 없음\n"
            "5점: 보통\n"
            "10점: 매우 관련 있음\n\n"
            "JSON 형식으로 답변하세요:\n"
            "{\"topic_relevance\": 점수}\n\n"
            "평가:"
        )

    def _create_novelty_prompt(self, text: str, recent_texts: List[str]) -> str:
        context_lines = "\n".join(f"- {t}" for t in recent_texts) if recent_texts else "(이전 발언 없음)"
        return (
            f"{GEMMA_SYSTEM_PROMPT}\n\n"
            "최근 회의 발언들:\n"
            f"{context_lines}\n\n"
            f'새로운 발언: "{text}"\n\n'
            "이 발언이 최근 발언들과 비교하여 얼마나 새로운 정보나 관점을 제공하는지 0-10점으로 평가하세요.\n"
            "0점: 완전한 반복\n"
            "5점: 부분적으로 새로움\n"
            "10점: 완전히 새로운 정보/관점\n\n"
            "JSON 형식으로 답변하세요:\n"
            "{\"novelty\": 점수}\n\n"
            "평가:"
        )

    def _query_model(self, prompt: str, score_key: str, default: float) -> float:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data.get("response", "")
            return self._extract_score(raw_text, score_key, default)
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            logger.error("Ollama 요청 실패: %s", exc)
            return default
        except ValueError as exc:
            logger.error("Ollama 응답 파싱 실패: %s", exc)
            return default

    def _extract_score(self, response_text: str, score_key: str, default: float) -> float:
        json_match = re.search(r"\{[^}]*\}", response_text)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                if score_key in payload:
                    return float(payload[score_key])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        number_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
        if number_match:
            try:
                value = float(number_match.group(1))
                return max(0.0, min(10.0, value))
            except ValueError:
                pass
        return default

OLLAMA_MODEL = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma3:1B")
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
    with meeting_state_lock:
        payload = {
            "type": "session_status",
            "is_active": meeting_state.is_active,
            "session_id": meeting_state.session_id,
            "topic": meeting_state.topic,
            "expected_speakers": meeting_state.expected_speakers,
            "speaker_id": meeting_state.speaker_id,
        }
    enqueue_event(payload)


def handle_transcription(text: str) -> None:
    cleaned = text.strip()
    if not cleaned:
        return
    with meeting_state_lock:
        if not meeting_state.is_active:
            return
        speaker_id = meeting_state.speaker_id
        session_id = meeting_state.session_id
    logger.debug("새 전사: %s", cleaned)
    enqueue_event(
        {
            "type": "transcription",
            "text": cleaned,
            "speaker_id": speaker_id,
            "timestamp": datetime.now().isoformat(),
        }
    )
    schedule_evaluation(cleaned, speaker_id, session_id)


def schedule_evaluation(text: str, speaker_id: str, session_id: Optional[str]) -> None:
    loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "event_loop", None)
    queue_: Optional[asyncio.Queue] = getattr(app.state, "evaluation_queue", None)
    if loop is None or queue_ is None:
        logger.warning("평가 큐가 준비되지 않았습니다")
        return
    payload = {
        "text": text,
        "speaker_id": speaker_id,
        "session_id": session_id,
    }
    asyncio.run_coroutine_threadsafe(queue_.put(payload), loop)


def ensure_diarization_service(device: str, max_speakers: int) -> DiarizationService:
    """선택한 장치와 화자 수 설정에 맞는 다이어리제이션 서비스를 준비한다."""

    global diarization_service
    target_speakers = max(1, min(int(max_speakers), MAX_SPEAKERS))
    if diarization_service:
        same_device = diarization_service.encoder.device == device
        same_limit = getattr(diarization_service, "max_speakers", MAX_SPEAKERS) == target_speakers
        if same_device and same_limit:
            return diarization_service
        diarization_service.stop()
    diarization_service = DiarizationService(device=device, max_speakers=target_speakers)
    return diarization_service


def handle_diarization_segment(segment: DiarizationSegment) -> None:
    if not meeting_state.is_active:
        return
    service = diarization_service
    offset = 0.0
    if service is not None:
        offset = max(0.0, segment.start_time - service.timeline_start)
    duration = max(0.0, segment.end_time - segment.start_time)
    enqueue_event(
        {
            "type": "diarization",
            "speaker_id": segment.speaker_label,
            "is_pending": segment.is_pending,
            "similarity": round(float(segment.similarity), 3),
            "offset": offset,
            "duration": duration,
        }
    )


def update_active_speaker(speaker_label: str) -> None:
    with meeting_state_lock:
        if not meeting_state.is_active:
            return
        previous = meeting_state.speaker_id
        meeting_state.speaker_id = speaker_label
    if speaker_label == previous or speaker_label == "Pending":
        return
    enqueue_event(
        {
            "type": "active_speaker",
            "speaker_id": speaker_label,
            "timestamp": datetime.now().isoformat(),
        }
    )


async def _evaluation_worker() -> None:
    queue_: asyncio.Queue = app.state.evaluation_queue
    loop = asyncio.get_running_loop()
    while True:
        task = await queue_.get()
        try:
            await loop.run_in_executor(
                None,
                lambda t=task: _evaluate_and_broadcast(
                    t.get("text", ""),
                    t.get("speaker_id", "Speaker 1"),
                    t.get("session_id"),
                ),
            )
        finally:
            queue_.task_done()


def _evaluate_and_broadcast(text: str, speaker_id: str, session_id: Optional[str]) -> None:
    if not text or session_id is None:
        return
    if not meeting_state.is_active or meeting_state.session_id != session_id:
        return
    try:
        result = ollama_evaluator.evaluate(text, speaker_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("평가 실패: %s", exc)
        result = {"topic_relevance": 5.0, "novelty": 5.0}

    meeting_stats.add_statement(speaker_id, result["topic_relevance"], result["novelty"])
    with meeting_history_lock:
        meeting_history.append(
            {
                "text": text,
                "speaker_id": speaker_id,
                "topic_relevance": result["topic_relevance"],
                "novelty": result["novelty"],
                "timestamp": datetime.now().isoformat(),
            }
        )

    enqueue_event(
        {
            "type": "evaluation",
            "speaker_id": speaker_id,
            "topic_relevance": result["topic_relevance"],
            "novelty": result["novelty"],
            "timestamp": datetime.now().isoformat(),
        }
    )

    enqueue_event(
        {
            "type": "stats_update",
            "overall_stats": meeting_stats.overall_dict(),
            "speaker_stats": meeting_stats.speaker_dict(),
        }
    )


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
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
                <div>
                    <h1 class="text-3xl font-bold text-gray-800 mb-2">🎯 MeetingProgram</h1>
                    <p class="text-gray-600">실시간 회의 평가 시스템 (Whisper + Gemma 1B)</p>
                </div>
                <div class="w-full md:w-auto">
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                        <input type="text" id="meetingTopic" placeholder="회의 주제" class="p-2 border rounded" required>
                        <input type="number" id="expectedSpeakers" min="1" max="10" value="2" class="p-2 border rounded" placeholder="예상 화자 수">
                        <input type="hidden" id="initialSpeaker" value="Speaker 1">
                    </div>
                    <div class="flex flex-wrap gap-3">
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
            </div>
        </header>

        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">음성 타임라인</h2>
            <div class="space-y-4">
                <canvas id="waveformCanvas" class="w-full h-40 rounded bg-gray-100"></canvas>
                <canvas id="speakerCanvas" class="w-full h-20 rounded bg-gray-100"></canvas>
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

        const waveformCanvas = document.getElementById('waveformCanvas');
        const speakerCanvas = document.getElementById('speakerCanvas');
        let waveformPoints = [];
        let waveformWindowSeconds = 0;
        let speakerSegments = [];
        const timelineWindowSeconds = 120;
        const speakerPalette = ['#2563eb', '#ef4444', '#10b981', '#f97316', '#8b5cf6', '#ec4899', '#22d3ee', '#f59e0b'];
        const speakerColors = { Pending: '#9ca3af', '대기 중': '#9ca3af' };
        let speakerPaletteIndex = 0;

        window.addEventListener('resize', () => {
            drawWaveform();
            drawSpeakerTimeline();
        });

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const reportBtn = document.getElementById('reportBtn');

        function getSpeakerColor(label) {
            if (!label) {
                return '#2563eb';
            }
            if (!speakerColors[label]) {
                const color = speakerPalette[speakerPaletteIndex % speakerPalette.length];
                speakerColors[label] = color;
                speakerPaletteIndex += 1;
            }
            return speakerColors[label];
        }

        function prepareCanvas(canvas) {
            if (!canvas) return null;
            const width = canvas.clientWidth || canvas.parentElement.clientWidth || 0;
            const height = canvas.clientHeight || 0;
            const ratio = window.devicePixelRatio || 1;
            const scaledWidth = Math.max(1, Math.floor(width * ratio));
            const scaledHeight = Math.max(1, Math.floor(height * ratio));
            if (canvas.width !== scaledWidth || canvas.height !== scaledHeight) {
                canvas.width = scaledWidth;
                canvas.height = scaledHeight;
            }
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            return { ctx, width, height };
        }

        function drawWaveform() {
            if (!waveformCanvas) return;
            const prepared = prepareCanvas(waveformCanvas);
            if (!prepared) return;
            const { ctx, width, height } = prepared;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = '#e5e7eb';
            ctx.fillRect(0, 0, width, height);

            if (!waveformPoints.length) {
                ctx.fillStyle = '#6b7280';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('오디오 파형 대기 중', width / 2, height / 2);
                return;
            }

            const mid = height / 2;
            ctx.strokeStyle = '#1d4ed8';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < waveformPoints.length; i += 1) {
                const x = (i / (waveformPoints.length - 1)) * width;
                const amplitude = waveformPoints[i];
                const y1 = mid - amplitude * (mid - 2);
                const y2 = mid + amplitude * (mid - 2);
                ctx.moveTo(x, y1);
                ctx.lineTo(x, y2);
            }
            ctx.stroke();

            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(0, mid);
            ctx.lineTo(width, mid);
            ctx.stroke();
        }

        function drawSpeakerTimeline() {
            if (!speakerCanvas) return;
            const prepared = prepareCanvas(speakerCanvas);
            if (!prepared) return;
            const { ctx, width, height } = prepared;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = '#e5e7eb';
            ctx.fillRect(0, 0, width, height);

            if (!speakerSegments.length) {
                ctx.fillStyle = '#6b7280';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('화자 데이터 대기 중', width / 2, height / 2);
                return;
            }

            let latestEnd = 0;
            speakerSegments.forEach((seg) => {
                latestEnd = Math.max(latestEnd, seg.start + seg.duration);
            });
            const activeWindow = Math.max(10, waveformWindowSeconds || timelineWindowSeconds);
            const windowStart = Math.max(0, latestEnd - activeWindow);
            const windowEnd = windowStart + activeWindow;
            const barHeight = Math.max(6, height * 0.5);
            const barTop = (height - barHeight) / 2;

            ctx.fillStyle = '#d1d5db';
            ctx.fillRect(0, barTop, width, barHeight);

            speakerSegments.forEach((seg) => {
                const segStart = Math.max(seg.start, windowStart);
                const segEnd = Math.min(seg.start + seg.duration, windowEnd);
                if (segEnd <= segStart) {
                    return;
                }
                const startX = ((segStart - windowStart) / activeWindow) * width;
                const segmentWidth = Math.max(2, ((segEnd - segStart) / activeWindow) * width);
                const color = seg.pending ? '#9ca3af' : getSpeakerColor(seg.label);
                ctx.fillStyle = color;
                ctx.fillRect(startX, barTop, segmentWidth, barHeight);
                if (!seg.pending && segmentWidth > 40) {
                    ctx.fillStyle = '#1f2937';
                    ctx.font = '12px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(seg.label, startX + segmentWidth / 2, barTop + barHeight / 2);
                }
            });
        }

        function updateWaveform(data) {
            waveformPoints = Array.isArray(data.points) ? data.points : [];
            waveformWindowSeconds = typeof data.window_seconds === 'number' ? data.window_seconds : 0;
            drawWaveform();
            drawSpeakerTimeline();
        }

        function trimSpeakerSegments() {
            if (!speakerSegments.length) return;
            let latestEnd = 0;
            speakerSegments.forEach((seg) => {
                latestEnd = Math.max(latestEnd, seg.start + seg.duration);
            });
            const windowSize = Math.max(timelineWindowSeconds, waveformWindowSeconds || timelineWindowSeconds);
            const cutoff = Math.max(0, latestEnd - windowSize);
            speakerSegments = speakerSegments.filter((seg) => seg.start + seg.duration >= cutoff);
        }

        function handleDiarization(data) {
            showCurrentSpeaker(data);
            if (typeof data.offset !== 'number') {
                drawSpeakerTimeline();
                return;
            }
            const segment = {
                start: Math.max(0, data.offset),
                duration: Math.max(0, data.duration || 0),
                label: data.speaker_id || 'Unknown',
                pending: !!data.is_pending,
            };
            if (segment.duration <= 0) {
                drawSpeakerTimeline();
                return;
            }
            speakerSegments.push(segment);
            trimSpeakerSegments();
            drawSpeakerTimeline();
        }

        drawWaveform();
        drawSpeakerTimeline();

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
                case 'waveform':
                    updateWaveform(data);
                    break;
                case 'diarization':
                    handleDiarization(data);
                    break;
            }
        }

        function updateSessionStatus(data) {
            const status = document.getElementById('status');
            const expectedField = document.getElementById('expectedSpeakers');
            const initialField = document.getElementById('initialSpeaker');
            if (expectedField && typeof data.expected_speakers === 'number') {
                expectedField.value = data.expected_speakers;
            }
            if (data.is_active) {
                if (data.session_id && data.session_id !== currentSession) {
                    waveformPoints = [];
                    speakerSegments = [];
                    Object.keys(speakerColors).forEach((key) => {
                        if (key !== 'Pending' && key !== '대기 중') {
                            delete speakerColors[key];
                        }
                    });
                    speakerPaletteIndex = 0;
                    drawWaveform();
                    drawSpeakerTimeline();
                }
                status.textContent = `🎤 회의 진행 중 (주제: ${data.topic})`;
                status.className = 'text-green-600 font-semibold';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                reportBtn.disabled = true;
                if (initialField && typeof data.speaker_id === 'string') {
                    initialField.value = data.speaker_id;
                }
                currentSession = data.session_id;
            } else {
                status.textContent = '⏸️ 대기 중';
                status.className = 'text-gray-600';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                reportBtn.disabled = true;
                waveformPoints = [];
                speakerSegments = [];
                Object.keys(speakerColors).forEach((key) => {
                    if (key !== 'Pending' && key !== '대기 중') {
                        delete speakerColors[key];
                    }
                });
                speakerPaletteIndex = 0;
                drawWaveform();
                drawSpeakerTimeline();
                if (initialField) {
                    initialField.value = 'Speaker 1';
                }
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

        function showCurrentSpeaker(data) {
            const el = document.getElementById('currentSpeaker');
            if (!el) return;
            const label = data.speaker_id || 'Unknown';
            const isPending = !!data.is_pending;
            const similarity = typeof data.similarity === 'number' ? data.similarity : null;
            const offset = typeof data.offset === 'number' ? data.offset : null;
            const parts = [`현재 화자: ${label}`];
            if (similarity !== null) {
                parts.push(`유사도 ${similarity.toFixed(2)}`);
            }
            if (offset !== null) {
                parts.push(`+${offset.toFixed(1)}초`);
            }
            parts.push(isPending ? '상태: 분류 중' : '상태: 확정');
            el.textContent = parts.join(' | ');
            el.className = 'mt-2 text-sm font-semibold';
            el.style.color = isPending ? '#ca8a04' : getSpeakerColor(label);
        }

        function addTranscription(data) {
            const feed = document.getElementById('transcriptionFeed');
            const item = document.createElement('div');
            const color = getSpeakerColor(data.speaker_id);
            item.className = 'border-l-4 pl-4 py-2';
            item.style.borderLeft = `4px solid ${color}`;
            item.innerHTML = `
                <div class="flex justify-between">
                    <span class="font-semibold speaker-label">${data.speaker_id}</span>
                    <span class="text-sm text-gray-500">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="mt-1">${data.text}</div>
            `;
            const labelEl = item.querySelector('.speaker-label');
            if (labelEl) {
                labelEl.style.color = color;
            }

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
            const speakerInput = document.getElementById('initialSpeaker');
            const expectedInput = document.getElementById('expectedSpeakers');
            const rawExpected = expectedInput ? parseInt(expectedInput.value, 10) : 2;
            const expectedSpeakers = Math.min(Math.max(rawExpected || 2, 1), 10);
            const initialSpeaker = speakerInput && speakerInput.value.trim() ? speakerInput.value.trim() : 'Speaker 1';
            if (!topic) {
                alert('회의 주제를 입력하세요.');
                return;
            }
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic, speaker_id: initialSpeaker, expected_speakers: expectedSpeakers })
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


@app.post("/api/start")
async def start_meeting(payload: Dict[str, Any]) -> Dict[str, Any]:
    if meeting_state.is_active:
        raise HTTPException(status_code=400, detail="이미 회의가 진행 중입니다")

    topic = (payload.get("topic") or "").strip()
    speaker_id = (payload.get("speaker_id") or "Speaker 1").strip() or "Speaker 1"
    expected_speakers_raw = payload.get("expected_speakers")
    try:
        expected_speakers = int(expected_speakers_raw)
    except (TypeError, ValueError):
        expected_speakers = 2
    expected_speakers = max(1, min(expected_speakers, MAX_SPEAKERS))
    if not topic:
        raise HTTPException(status_code=400, detail="회의 주제를 입력하세요")

    session_id = str(uuid.uuid4())[:8]
    with meeting_state_lock:
        meeting_state.session_id = session_id
        meeting_state.topic = topic
        meeting_state.speaker_id = speaker_id
        meeting_state.expected_speakers = expected_speakers

    meeting_stats.reset()
    ollama_evaluator.initialize(topic)
    global last_report_path, last_summary_payload
    with meeting_history_lock:
        meeting_history.clear()
    last_report_path = None
    last_summary_payload = None

    loop = asyncio.get_running_loop()
    diar_service: Optional[DiarizationService] = None
    try:
        await loop.run_in_executor(None, transcription_service.initialize)
        diar_service = await loop.run_in_executor(
            None,
            lambda: ensure_diarization_service(
                transcription_service.device,
                expected_speakers,
            ),
        )
        diar_service.set_segment_callback(handle_diarization_segment)
        diar_service.set_active_speaker_callback(update_active_speaker)
        diar_service.start()
        transcription_service.attach_diarization(diar_service)
        await loop.run_in_executor(None, lambda: transcription_service.start(handle_transcription))
    except Exception as exc:  # noqa: BLE001
        with meeting_state_lock:
            meeting_state.session_id = None
            meeting_state.topic = ""
            meeting_state.speaker_id = "Speaker 1"
            meeting_state.expected_speakers = 2
        if diar_service is not None:
            await loop.run_in_executor(None, diar_service.stop)
        transcription_service.attach_diarization(None)
        logger.exception("Whisper 전사 시작 실패: %s", exc)
        raise HTTPException(status_code=500, detail="전사 시작에 실패했습니다. 로그를 확인하세요.")

    with meeting_state_lock:
        meeting_state.is_active = True
    broadcast_session_status()
    enqueue_event(
        {
            "type": "stats_update",
            "overall_stats": meeting_stats.overall_dict(),
            "speaker_stats": meeting_stats.speaker_dict(),
        }
    )
    logger.info("회의 세션 시작: %s (주제=%s)", session_id, topic)

    return {"success": True, "session_id": session_id, "topic": topic}


@app.post("/api/stop/{session_id}")
async def stop_meeting(session_id: str) -> Dict[str, Any]:
    if not meeting_state.is_active or meeting_state.session_id != session_id:
        raise HTTPException(status_code=400, detail="진행 중인 세션이 없습니다")

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

    diar_service = diarization_service
    if diar_service is not None:
        await loop.run_in_executor(None, diar_service.stop)
    transcription_service.attach_diarization(None)

    with meeting_state_lock:
        meeting_state.is_active = False
    broadcast_session_status()
    enqueue_event(
        {
            "type": "diarization",
            "speaker_id": "대기 중",
            "is_pending": False,
            "similarity": 0.0,
            "offset": 0.0,
            "duration": 0.0,
        }
    )
    enqueue_event(
        {
            "type": "waveform",
            "points": [],
            "window_seconds": 0.0,
        }
    )

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

    logger.info("회의 세션 종료: %s", session_id)

    with meeting_state_lock:
        meeting_state.session_id = None
        meeting_state.topic = ""
        meeting_state.speaker_id = "Speaker 1"
        meeting_state.expected_speakers = 2

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
