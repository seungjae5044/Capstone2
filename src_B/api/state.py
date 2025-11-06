"""
Global state management for the meeting application.

This module manages all global state including:
- Meeting session state
- Statistics tracking
- Service instances (transcription, diarization, evaluation)
- Event broadcasting
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src_B.models.data_models import MeetingState, MeetingStatistics, TranscribedSegment
from src_B.transcription.service import TranscriptionService
from src_B.diarization.service import DiarizationService, DiarizationSegment, MAX_SPEAKERS
from src_B.evaluation.evaluator import OllamaEvaluator

logger = logging.getLogger(__name__)

# ============================================================================
# Global State Variables
# ============================================================================

# Configuration paths
CONFIG_PATH = Path("config_whisper.json")
REPORTS_DIR = Path("output") / "reports"

# Service instances
transcription_service = TranscriptionService(CONFIG_PATH)
diarization_service: Optional[DiarizationService] = None
OLLAMA_MODEL = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma3-270m-local-e3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_evaluator = OllamaEvaluator(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# Meeting state
meeting_state = MeetingState()
meeting_state_lock = threading.Lock()

# Statistics
meeting_stats = MeetingStatistics()

# Meeting history
meeting_history_lock = threading.Lock()
meeting_history: List[Dict[str, Any]] = []

# Report tracking
last_report_path: Optional[Path] = None
last_summary_payload: Optional[Dict[str, Any]] = None

# Deduplication tracking for transcriptions
_last_transcription_text: Optional[str] = None
_last_transcription_speaker: Optional[str] = None
_last_transcription_ts: float = 0.0

# Event loop reference (will be set by app startup)
_app_state: Optional[Any] = None


# ============================================================================
# App State Management
# ============================================================================

def set_app_state(app_state: Any) -> None:
    """Set the FastAPI app.state reference for accessing event loop and queues."""
    global _app_state
    _app_state = app_state


def get_app_state() -> Optional[Any]:
    """Get the FastAPI app.state reference."""
    return _app_state


# ============================================================================
# Event Broadcasting Helpers
# ============================================================================

def enqueue_event(message: Dict[str, Any]) -> None:
    """Queue a message for async broadcast to WebSocket clients."""
    if _app_state is None:
        logger.warning("App state not initialized, cannot enqueue event")
        return

    loop: Optional[asyncio.AbstractEventLoop] = getattr(_app_state, "event_loop", None)
    queue_: Optional[asyncio.Queue] = getattr(_app_state, "event_queue", None)

    if loop is None or queue_ is None:
        logger.warning("Event loop or queue not ready")
        return

    asyncio.run_coroutine_threadsafe(queue_.put(message), loop)


def broadcast_session_status() -> None:
    """Broadcast current session status to all connected clients."""
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


def build_speaker_stats_payload(
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build a comprehensive speaker statistics payload."""
    speaker_summaries = transcription_service.speaker_summaries()
    total_duration = sum(item["duration"] for item in speaker_summaries) or 0.0
    speakers_payload: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # Add speakers with transcription summaries
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

    # Add speakers that only have stats (no transcription summary)
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
    """Broadcast current statistics to all connected clients."""
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


# ============================================================================
# Transcription Handlers
# ============================================================================

def handle_transcription(segment: TranscribedSegment) -> None:
    """Handle a new transcribed segment from the transcription service."""
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

    logger.debug("New transcription: %s", cleaned)

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
    """Schedule evaluation of a transcribed segment."""
    if _app_state is None:
        logger.warning("App state not initialized, cannot schedule evaluation")
        return

    loop: Optional[asyncio.AbstractEventLoop] = getattr(_app_state, "event_loop", None)
    queue_: Optional[asyncio.Queue] = getattr(_app_state, "evaluation_queue", None)

    if loop is None or queue_ is None:
        logger.warning("Evaluation queue not ready")
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


# ============================================================================
# Diarization Handlers
# ============================================================================

def ensure_diarization_service(device: str, max_speakers: int) -> DiarizationService:
    """Ensure diarization service is initialized with correct device and speaker count."""
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
    """Handle a diarization segment from the diarization service."""
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
    """Update the currently active speaker."""
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
