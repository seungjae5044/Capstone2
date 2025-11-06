"""Real-time speaker diarization service.

This module provides the DiarizationService class which processes audio streams
in real-time to identify and track different speakers.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np
from scipy.signal import resample_poly

from src_B.diarization.constants import (
    MAX_SPEAKERS,
    SAMPLE_RATE,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
)
from src_B.diarization.speaker_handler import SpeakerHandler
from src_B.models.data_models import DiarizationSegment, SpeakerDecision
from src_B.models.ml_models import SileroVAD, SpeechBrainEncoder


class DiarizationService:
    """Processes audio chunks to emit diarization segments.

    This service manages a real-time audio processing pipeline that:
    1. Accepts audio chunks via add_audio()
    2. Buffers audio into fixed-size windows
    3. Detects speech using VAD (Voice Activity Detection)
    4. Extracts speaker embeddings from speech segments
    5. Classifies speakers and emits diarization segments

    The service runs a background thread that processes queued audio chunks
    and invokes callbacks when speakers change or segments are identified.

    Attributes:
        sample_rate: Target audio sample rate in Hz
        window_seconds: Duration of analysis window in seconds
        stride_seconds: Time between consecutive windows in seconds
        vad: Voice activity detection model
        encoder: Speaker embedding extraction model
        max_speakers: Maximum number of concurrent speakers
        speakers: Speaker management and classification handler
        segment_callback: Optional callback for emitting diarization segments
        active_speaker_callback: Optional callback for active speaker changes
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = WINDOW_SECONDS,
        stride_seconds: float = STRIDE_SECONDS,
        device: str = "cpu",
        max_speakers: int = MAX_SPEAKERS,
    ) -> None:
        """Initialize the diarization service.

        Args:
            sample_rate: Target sample rate for audio processing (default: SAMPLE_RATE)
            window_seconds: Duration of analysis window (default: WINDOW_SECONDS)
            stride_seconds: Time between windows (default: STRIDE_SECONDS)
            device: Device for ML models ('cpu' or 'cuda')
            max_speakers: Maximum number of speakers to track (default: MAX_SPEAKERS)
        """
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.window_samples = int(sample_rate * window_seconds)
        self.stride_samples = max(1, int(sample_rate * stride_seconds))
        self.vad = SileroVAD()
        self.encoder = SpeechBrainEncoder(device=device)
        clamped = max(1, int(max_speakers))
        self.max_speakers = min(clamped, MAX_SPEAKERS)
        self.speakers = SpeakerHandler(max_speakers=self.max_speakers)

        self.segment_callback: Optional[Callable[[DiarizationSegment], None]] = None
        self.active_speaker_callback: Optional[Callable[[str], None]] = None

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.buf_idx = 0
        self.buffer_full = False
        self.samples_since_window = 0
        self.total_samples = 0
        self.timeline_start = 0.0
        self.pending_segments: List[DiarizationSegment] = []

        self._lock = threading.Lock()
        self._last_label: Optional[str] = None
        self._last_pending: Optional[bool] = None

    def set_segment_callback(self, callback: Callable[[DiarizationSegment], None]) -> None:
        """Set callback function for diarization segment emissions.

        Args:
            callback: Function to call with each DiarizationSegment
        """
        self.segment_callback = callback

    def set_active_speaker_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for active speaker changes.

        Args:
            callback: Function to call with speaker label when active speaker changes
        """
        self.active_speaker_callback = callback

    def start(self) -> None:
        """Start the diarization service and begin processing audio.

        Initializes the background worker thread and resets all internal state.
        """
        with self._lock:
            if self.worker and self.worker.is_alive():
                return
            self.stop_event.clear()
            self.audio_queue = queue.Queue(maxsize=200)
            self.buffer.fill(0)
            self.buf_idx = 0
            self.buffer_full = False
            self.samples_since_window = 0
            self.total_samples = 0
            self.timeline_start = time.time()
            self.pending_segments.clear()
            self.speakers.reset()
            self._last_label = None
            self._last_pending = None
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def stop(self) -> None:
        """Stop the diarization service and terminate the worker thread."""
        with self._lock:
            self.stop_event.set()
            if self.audio_queue:
                try:
                    self.audio_queue.put_nowait(np.array([], dtype=np.float32))
                except queue.Full:
                    pass
            if self.worker and self.worker.is_alive():
                self.worker.join(timeout=2.0)
            self.worker = None

    def add_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Add audio chunk to processing queue.

        Audio will be resampled if necessary and queued for processing by the
        background worker thread.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate of the input audio
        """
        if self.worker is None or not self.worker.is_alive():
            return
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if sample_rate != self.sample_rate:
            audio = resample_poly(audio, self.sample_rate, sample_rate)
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            pass

    def _run(self) -> None:
        """Background worker thread main loop.

        Continuously processes audio chunks from the queue until stop is signaled.
        """
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk.size == 0 and self.stop_event.is_set():
                break
            if chunk.size == 0:
                continue
            self._process_chunk(chunk)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Process an audio chunk by buffering and extracting windows.

        Args:
            chunk: Audio samples to process
        """
        chunk = chunk.astype(np.float32)
        self._append_to_buffer(chunk)
        self.total_samples += len(chunk)
        self.samples_since_window += len(chunk)
        while self.samples_since_window >= self.stride_samples:
            self.samples_since_window -= self.stride_samples
            window = self._get_window()
            if window is None:
                continue
            self._process_window(window)

    def _append_to_buffer(self, chunk: np.ndarray) -> None:
        """Append audio samples to circular buffer.

        Args:
            chunk: Audio samples to append
        """
        if len(chunk) >= self.window_samples:
            self.buffer[:] = chunk[-self.window_samples :]
            self.buf_idx = 0
            self.buffer_full = True
            return
        end_space = self.window_samples - self.buf_idx
        if len(chunk) < end_space:
            self.buffer[self.buf_idx : self.buf_idx + len(chunk)] = chunk
            self.buf_idx += len(chunk)
            if self.buf_idx >= self.window_samples:
                self.buf_idx = 0
                self.buffer_full = True
            return
        self.buffer[self.buf_idx :] = chunk[:end_space]
        remaining = len(chunk) - end_space
        if remaining > 0:
            self.buffer[:remaining] = chunk[end_space:]
        self.buf_idx = remaining
        self.buffer_full = True

    def _get_window(self) -> Optional[np.ndarray]:
        """Extract current analysis window from circular buffer.

        Returns:
            Window samples or None if buffer not yet full
        """
        if not self.buffer_full:
            return None
        if self.buf_idx == 0:
            return self.buffer.copy()
        window = np.empty(self.window_samples, dtype=np.float32)
        head = self.window_samples - self.buf_idx
        window[:head] = self.buffer[self.buf_idx :]
        window[head:] = self.buffer[: self.buf_idx]
        return window

    def _process_window(self, window: np.ndarray) -> None:
        """Process an analysis window to detect and classify speakers.

        Args:
            window: Audio window to analyze
        """
        is_speech = self.vad.is_speech(window, sample_rate=self.sample_rate)
        window_end_time = self.timeline_start + self.total_samples / self.sample_rate
        window_start_time = window_end_time - self.window_seconds

        if not is_speech:
            return

        embedding = self.encoder.embed(window, sample_rate=self.sample_rate)
        decision = self.speakers.classify(embedding, window_start_time)

        if decision.promoted_speaker_index is not None:
            label = self._speaker_label(decision.promoted_speaker_index)
            for pending_segment in self.pending_segments:
                pending_segment.speaker_label = label
                pending_segment.is_pending = False
                self._emit_segment(pending_segment)
            self.pending_segments.clear()
            if decision.speaker_index is None:
                decision = SpeakerDecision(
                    speaker_index=decision.promoted_speaker_index,
                    similarity=decision.similarity,
                    is_pending=False,
                )
            if self.active_speaker_callback:
                self.active_speaker_callback(label)

        if decision.speaker_index is None and decision.is_pending:
            segment = DiarizationSegment(
                start_time=window_start_time,
                end_time=window_end_time,
                speaker_label="Pending",
                similarity=decision.similarity,
                is_pending=True,
            )
            self.pending_segments.append(segment)
            self._emit_segment(segment)
            if self.active_speaker_callback:
                self.active_speaker_callback("Pending")
            return

        if decision.speaker_index is None:
            return

        label = self._speaker_label(decision.speaker_index)
        segment = DiarizationSegment(
            start_time=window_start_time,
            end_time=window_end_time,
            speaker_label=label,
            similarity=decision.similarity,
            is_pending=False,
        )
        self._emit_segment(segment)
        if self.active_speaker_callback:
            self.active_speaker_callback(label)

    def _speaker_label(self, speaker_index: int) -> str:
        """Convert speaker index to human-readable label.

        Args:
            speaker_index: Zero-based speaker index

        Returns:
            Speaker label string (e.g., "Speaker 1")
        """
        return f"Speaker {speaker_index + 1}"

    def _emit_segment(self, segment: DiarizationSegment) -> None:
        """Emit a diarization segment via callback if configured.

        Deduplicates consecutive segments with the same speaker and pending status.

        Args:
            segment: Segment to emit
        """
        if self.segment_callback:
            if (
                segment.speaker_label == self._last_label
                and segment.is_pending == self._last_pending
            ):
                return
            self._last_label = segment.speaker_label
            self._last_pending = segment.is_pending
            try:
                self.segment_callback(segment)
            except Exception:
                pass
