"""Data models for the meeting transcription and diarization system."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription model."""
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
    """Represents a transcribed speech segment."""
    text: str
    speaker_id: str
    speaker_name: str
    similarity: float
    start_time: datetime
    end_time: datetime
    duration: float


@dataclass
class SpeakerProfile:
    """Tracks speaker identity and statistics."""
    speaker_id: str
    display_name: str
    embedding: np.ndarray
    statement_count: int = 0
    duration: float = 0.0
    last_similarity: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_embedding(self, embedding: np.ndarray, alpha: float) -> None:
        """Update speaker embedding with exponential moving average."""
        if not np.any(self.embedding):
            self.embedding = embedding
        else:
            self.embedding = (1.0 - alpha) * self.embedding + alpha * embedding
        self.embedding = self.embedding / (np.linalg.norm(self.embedding) + 1e-8)


@dataclass
class TimelineSegment:
    """Timeline visualization data."""
    speaker_id: str
    speaker_name: str
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": (self.end_time - self.start_time).total_seconds(),
        }


@dataclass
class SpeakerDecision:
    """Represents speaker classification decision for diarization."""
    speaker_index: Optional[int]
    similarity: float
    is_pending: bool
    promoted_speaker_index: Optional[int] = None


@dataclass
class DiarizationSegment:
    """Represents a diarization time segment."""
    start_time: float
    end_time: float
    speaker_label: str
    similarity: float
    is_pending: bool


@dataclass
class MeetingState:
    """Global meeting session state."""
    session_id: Optional[str] = None
    last_session_id: Optional[str] = None
    topic: str = ""
    speaker_id: str = "Speaker 1"
    is_active: bool = False
    expected_speakers: int = 2


@dataclass
class SpeakerStatsEntry:
    """Per-speaker statistics accumulator."""
    total_statements: int = 0
    topic_sum: float = 0.0
    novelty_sum: float = 0.0

    def add(self, topic_score: float, novelty_score: float) -> None:
        """Add a statement's scores to the statistics."""
        self.total_statements += 1
        self.topic_sum += topic_score
        self.novelty_sum += novelty_score

    def avg_topic(self) -> float:
        """Calculate average topic relevance score."""
        return self.topic_sum / self.total_statements if self.total_statements else 0.0

    def avg_novelty(self) -> float:
        """Calculate average novelty score."""
        return self.novelty_sum / self.total_statements if self.total_statements else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_statements": self.total_statements,
            "avg_topic_relevance": self.avg_topic(),
            "avg_novelty": self.avg_novelty(),
        }


@dataclass
class MeetingStatistics:
    """Overall meeting statistics."""
    total_statements: int = 0
    topic_sum: float = 0.0
    novelty_sum: float = 0.0
    speakers: Dict[str, SpeakerStatsEntry] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> None:
        """Reset all statistics."""
        with self.lock:
            self.total_statements = 0
            self.topic_sum = 0.0
            self.novelty_sum = 0.0
            self.speakers.clear()

    def add_statement(self, speaker_id: str, topic_score: float, novelty_score: float) -> None:
        """Add a statement's scores to the meeting statistics."""
        with self.lock:
            self.total_statements += 1
            self.topic_sum += topic_score
            self.novelty_sum += novelty_score
            entry = self.speakers.setdefault(speaker_id, SpeakerStatsEntry())
            entry.add(topic_score, novelty_score)

    def overall_dict(self) -> Dict[str, Any]:
        """Get overall statistics as dictionary."""
        with self.lock:
            avg_topic = self.topic_sum / self.total_statements if self.total_statements else 0.0
            avg_novelty = self.novelty_sum / self.total_statements if self.total_statements else 0.0
            return {
                "total_statements": self.total_statements,
                "avg_topic_relevance": avg_topic,
                "avg_novelty": avg_novelty,
            }

    def speaker_dict(self, transcription_service=None) -> Dict[str, Any]:
        """Get per-speaker statistics as dictionary."""
        with self.lock:
            data = {speaker: stats.to_dict() for speaker, stats in self.speakers.items()}

        # Add duration from transcription service if available
        if transcription_service:
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
