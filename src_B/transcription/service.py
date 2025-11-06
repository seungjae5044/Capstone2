"""
Real-time speech transcription service using OpenAI Whisper.

This module provides the TranscriptionService class for performing real-time
speech-to-text transcription with speaker identification, voice activity detection,
and integration with diarization services.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timedelta
from math import gcd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src_B.config.config import DEFAULT_CONFIG, deep_update, load_config, select_device
from src_B.models.data_models import TranscribedSegment, WhisperConfig
from src_B.models.ml_models import SileroVAD, SpeechBrainEncoder
from src_B.transcription.speaker_handler import SpeakerHandler
from src_B.utils.audio import RollingWaveform
from src_B.utils.timeline import TimelineManager

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Real-time speech transcription service with speaker identification.

    This service manages the complete transcription pipeline including:
    - Audio capture from microphone using sounddevice
    - Voice Activity Detection (VAD) using Silero VAD
    - Speaker embedding extraction using SpeechBrain
    - Speech-to-text transcription using Whisper
    - Speaker identification and tracking
    - Timeline management for speaker segments
    - Waveform visualization data generation
    - Optional integration with diarization service

    The service runs transcription in a background thread and delivers results
    via callbacks. It supports both CPU and GPU acceleration and handles audio
    resampling automatically.

    Attributes:
        config_path: Path to configuration JSON file
        pipeline: Whisper transcription pipeline (initialized lazily)
        device: Compute device ('cpu', 'cuda', or 'mps')
        torch_dtype: PyTorch data type for model inference
        config: Whisper configuration object
        stop_event: Threading event for graceful shutdown
        thread: Background thread for transcription processing
        audio_queue: Queue for audio chunks from microphone
        segment_callback: Callback function for transcribed segments
        lock: Threading lock for initialization
        resample_up: Upsampling factor for audio resampling
        resample_down: Downsampling factor for audio resampling
        vad: Voice Activity Detection model
        encoder: Speaker embedding extraction model
        speakers: Speaker identification and tracking handler
        timeline: Timeline manager for speaker segments
        waveform_buffer: Rolling buffer for waveform visualization
        last_waveform_emit: Timestamp of last waveform event emission
        waveform_emit_interval: Interval in seconds between waveform events
        diarization_service: Optional diarization service for cross-validation
        waveform_callback: Optional callback for waveform visualization data

    Example:
        >>> from pathlib import Path
        >>> service = TranscriptionService(Path("config_whisper.json"))
        >>> service.initialize()
        >>>
        >>> def on_segment(segment):
        ...     print(f"{segment.speaker_name}: {segment.text}")
        >>>
        >>> service.start(callback=on_segment)
        >>> # Transcription runs in background...
        >>> service.stop()
    """

    def __init__(self, config_path: Path) -> None:
        """
        Initialize the TranscriptionService.

        Args:
            config_path: Path to the Whisper configuration JSON file

        Note:
            The actual model initialization is deferred until initialize() is called.
            This allows for faster startup and lazy loading of heavy ML models.
        """
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

        # Audio resampling parameters (computed during initialization)
        self.resample_up = 1
        self.resample_down = 1

        # ML models for VAD and speaker identification
        self.vad = SileroVAD()
        self.encoder = SpeechBrainEncoder()
        self.speakers = SpeakerHandler()
        self.timeline = TimelineManager()

        # Waveform visualization
        self.waveform_buffer = RollingWaveform()
        self.last_waveform_emit = 0.0
        self.waveform_emit_interval = 2.0

        # Optional diarization service integration
        self.diarization_service = None

        # Optional callback for waveform visualization events
        self.waveform_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def initialize(self) -> bool:
        """
        Initialize the Whisper transcription pipeline.

        This method loads the configuration, selects the appropriate compute device,
        loads the Whisper model, and sets up the transcription pipeline with all
        necessary parameters.

        The initialization is thread-safe and will only execute once even if called
        multiple times concurrently.

        Returns:
            True if initialization successful, False otherwise

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> if service.initialize():
            ...     print(f"Using device: {service.device}")
            Using device: cuda
        """
        try:
            with self.lock:
                if self.pipeline is not None:
                    return True

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
                return True

        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}", exc_info=True)
            return False

    def is_running(self) -> bool:
        """
        Check if transcription is currently running.

        Returns:
            True if transcription thread is active, False otherwise

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> print(f"Running: {service.is_running()}")
            Running: False
            >>> service.start(lambda seg: print(seg.text))
            >>> print(f"Running: {service.is_running()}")
            Running: True
        """
        return self.thread is not None and self.thread.is_alive()

    def reset_state(self) -> None:
        """
        Reset speaker profiles and timeline.

        This method clears all speaker tracking data and timeline segments,
        useful when starting a new meeting or session while keeping the
        transcription service initialized.

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> service.initialize()
            >>> # After some transcription...
            >>> service.reset_state()  # Clear for new session
        """
        self.speakers.reset()
        self.timeline.reset()

    def attach_diarization(self, service) -> None:
        """
        Attach a diarization service for cross-validation.

        This allows the transcription service to feed audio to a diarization
        service for speaker identification cross-validation and improved accuracy.

        Args:
            service: DiarizationService instance to attach

        Example:
            >>> from src_B.diarization.service import DiarizationService
            >>> transcription = TranscriptionService(Path("config_whisper.json"))
            >>> diarization = DiarizationService(expected_speakers=2)
            >>> transcription.attach_diarization(diarization)
        """
        self.diarization_service = service

    def start(self, callback: Callable[[TranscribedSegment], None]) -> None:
        """
        Start real-time transcription.

        This method starts the background transcription thread which captures audio
        from the default microphone, processes it through the pipeline, and delivers
        transcribed segments via the callback function.

        Args:
            callback: Function to call with each transcribed segment.
                     Receives a TranscribedSegment object as parameter.

        Raises:
            RuntimeError: If pipeline is not initialized or already running

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> service.initialize()
            >>>
            >>> def handle_segment(segment):
            ...     print(f"[{segment.speaker_name}] {segment.text}")
            ...     print(f"Duration: {segment.duration:.1f}s, Similarity: {segment.similarity:.2f}")
            >>>
            >>> service.start(callback=handle_segment)
            >>> # Let it run for a while...
            >>> service.stop()
        """
        if self.pipeline is None:
            raise RuntimeError("Whisper 파이프라인이 초기화되지 않았습니다")

        if self.is_running():
            raise RuntimeError("이미 전사가 진행 중입니다")

        self.segment_callback = callback
        self.stop_event.clear()
        self.reset_state()
        self.audio_queue = queue.Queue(maxsize=self.config.queue_maxsize if self.config else 100)
        self.waveform_buffer.reset()
        self.last_waveform_emit = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("실시간 전사 스레드 시작")

    def stop(self) -> None:
        """
        Stop real-time transcription.

        This method gracefully shuts down the transcription thread and cleans up
        resources. It waits up to 2 seconds for the thread to terminate.

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> service.initialize()
            >>> service.start(lambda seg: print(seg.text))
            >>> # After some time...
            >>> service.stop()
            >>> print(f"Still running: {service.is_running()}")
            Still running: False
        """
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

    def speaker_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summary statistics for all tracked speakers.

        Returns:
            List of dictionaries containing speaker statistics:
            - speaker_id: Unique identifier for the speaker
            - name: Display name of the speaker
            - count: Number of segments from this speaker
            - duration: Total speaking duration in seconds
            - last_similarity: Most recent similarity score
            - updated_at: Last update timestamp (ISO format)

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> # After some transcription...
            >>> for summary in service.speaker_summaries():
            ...     print(f"{summary['name']}: {summary['count']} segments, "
            ...           f"{summary['duration']:.1f}s total")
            Speaker 1: 5 segments, 23.4s total
            Speaker 2: 3 segments, 15.7s total
        """
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
        """
        Get a snapshot of the complete speaker timeline.

        Returns:
            List of timeline segments with speaker and timing information

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> # After some transcription...
            >>> timeline = service.timeline_snapshot()
            >>> for segment in timeline:
            ...     print(f"{segment['speaker_name']}: {segment['duration']:.1f}s")
            Speaker 1: 5.2s
            Speaker 2: 3.8s
        """
        return self.timeline.snapshot()

    def timeline_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent timeline segment.

        Returns:
            Dictionary with latest segment info, or None if no segments

        Example:
            >>> service = TranscriptionService(Path("config_whisper.json"))
            >>> # After some transcription...
            >>> latest = service.timeline_latest()
            >>> if latest:
            ...     print(f"Currently: {latest['speaker_name']}")
        """
        return self.timeline.latest()

    def _run(self) -> None:
        """
        Main transcription loop (runs in background thread).

        This method handles the complete transcription pipeline:
        1. Captures audio from microphone via sounddevice
        2. Resamples audio to target sample rate
        3. Feeds audio to diarization service if attached
        4. Updates waveform visualization buffer
        5. Detects speech using VAD
        6. Extracts speaker embeddings
        7. Classifies speaker and updates profiles
        8. Performs Whisper transcription
        9. Delivers results via callback

        The loop continues until stop_event is set.

        Note:
            This is an internal method and should not be called directly.
            Use start() and stop() instead.
        """
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
                    if self.diarization_service:
                        self.diarization_service.add_audio(resampled_chunk, target_sr)
                    self.waveform_buffer.append_chunk(resampled_chunk, target_sr)
                    now = time.time()
                    if now - self.last_waveform_emit >= self.waveform_emit_interval:
                        points = self.waveform_buffer.get_points()
                        if points and self.waveform_callback:
                            self.waveform_callback(
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
