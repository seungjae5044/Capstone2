"""Server-side real-time speaker diarization utilities."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
STRIDE_SECONDS = 0.1
VAD_THRESHOLD = 0.5
PENDING_THRESHOLD = 0.3
EMBEDDING_UPDATE_THRESHOLD = 0.4
MIN_PENDING_SIZE = 30
AUTO_CLUSTER_DISTANCE_THRESHOLD = 0.6
MIN_CLUSTER_SIZE = 15
MAX_SPEAKERS = 10


class SileroVAD:
    """Lightweight wrapper around Silero VAD."""

    def __init__(self, threshold: float = VAD_THRESHOLD) -> None:
        self.threshold = threshold
        self._model = None
        self._get_speech_ts = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False,
            )
        except Exception as exc:  # pragma: no cover - download guard
            raise RuntimeError("Silero VAD 모델을 불러오지 못했습니다.") from exc
        self._model = model.to("cpu")
        self._get_speech_ts = utils[0]

    def is_speech(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        if self._model is None or self._get_speech_ts is None or len(audio) < 400:
            return False
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        with torch.no_grad():
            speech_timestamps = self._get_speech_ts(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                sampling_rate=sample_rate,
                return_seconds=False,
            )
        return len(speech_timestamps) > 0


class SpeechBrainEncoder:
    """ECAPA-TDNN embedding extractor."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "speechbrain 패키지가 필요합니다. 'pip install speechbrain' 후 다시 시도하세요."
            ) from exc

        self.device = device
        self._encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self._encoder.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()


@dataclass
class SpeakerDecision:
    speaker_index: Optional[int]
    similarity: float
    is_pending: bool
    promoted_speaker_index: Optional[int] = None


@dataclass
class DiarizationSegment:
    start_time: float
    end_time: float
    speaker_label: str
    similarity: float
    is_pending: bool


class SpeakerHandler:
    """Maintains running speaker embeddings and pending queue."""

    def __init__(
        self,
        max_speakers: int = MAX_SPEAKERS,
        change_threshold: float = PENDING_THRESHOLD,
        min_pending: int = MIN_PENDING_SIZE,
    ) -> None:
        self.max_speakers = max_speakers
        self.change_threshold = change_threshold
        self.min_pending = min_pending
        self.mean_embs: List[Optional[np.ndarray]] = [None] * max_speakers
        self.spk_embs: List[List[np.ndarray]] = [[] for _ in range(max_speakers)]
        self.active_spks: set[int] = set()
        self.pending_embs: List[np.ndarray] = []
        self.pending_times: List[float] = []

    def reset(self) -> None:
        self.mean_embs = [None] * self.max_speakers
        self.spk_embs = [[] for _ in range(self.max_speakers)]
        self.active_spks.clear()
        self.pending_embs.clear()
        self.pending_times.clear()

    def classify(self, emb: np.ndarray, seg_start_time: float) -> SpeakerDecision:
        if not self.active_spks:
            if len(self.active_spks) < self.max_speakers:
                self.spk_embs[0].append(emb)
                self.mean_embs[0] = emb
                self.active_spks.add(0)
                return SpeakerDecision(speaker_index=0, similarity=1.0, is_pending=False)
            return SpeakerDecision(speaker_index=None, similarity=0.0, is_pending=True)

        active_mean_embs: List[np.ndarray] = []
        active_ids: List[int] = []
        for spk_id in self.active_spks:
            mean_emb = self.mean_embs[spk_id]
            if mean_emb is not None:
                active_mean_embs.append(mean_emb)
                active_ids.append(spk_id)

        if not active_mean_embs:
            self.spk_embs[0].append(emb)
            self.mean_embs[0] = emb
            self.active_spks.add(0)
            return SpeakerDecision(speaker_index=0, similarity=1.0, is_pending=False)

        emb_norm = emb / np.linalg.norm(emb)
        means = np.array(active_mean_embs)
        means_norm = means / np.linalg.norm(means, axis=1, keepdims=True)
        similarities = np.dot(means_norm, emb_norm)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_spk = active_ids[best_idx]

        if best_sim >= EMBEDDING_UPDATE_THRESHOLD:
            self.spk_embs[best_spk].append(emb)
            self.mean_embs[best_spk] = np.median(self.spk_embs[best_spk], axis=0)
            return SpeakerDecision(speaker_index=best_spk, similarity=best_sim, is_pending=False)

        if best_sim >= self.change_threshold:
            return SpeakerDecision(speaker_index=best_spk, similarity=best_sim, is_pending=False)

        if len(self.active_spks) < self.max_speakers:
            self.pending_embs.append(emb)
            self.pending_times.append(seg_start_time)
            promoted = self._maybe_promote_pending()
            return SpeakerDecision(
                speaker_index=None,
                similarity=best_sim,
                is_pending=True,
                promoted_speaker_index=promoted,
            )

        return SpeakerDecision(speaker_index=best_spk, similarity=best_sim, is_pending=False)

    def _maybe_promote_pending(self) -> Optional[int]:
        if len(self.pending_embs) < MIN_CLUSTER_SIZE:
            return None
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=AUTO_CLUSTER_DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(np.array(self.pending_embs))
            unique_labels = np.unique(labels)
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            target_cluster = max(cluster_sizes, key=cluster_sizes.get)
            largest = cluster_sizes[target_cluster]
            if largest < MIN_CLUSTER_SIZE:
                return None
            new_spk_id = self._next_speaker_id()
            if new_spk_id is None:
                return None
            cluster_embs = [self.pending_embs[i] for i, label in enumerate(labels) if label == target_cluster]
            self.spk_embs[new_spk_id] = list(cluster_embs)
            self.mean_embs[new_spk_id] = np.median(cluster_embs, axis=0)
            self.active_spks.add(new_spk_id)
            self.pending_embs.clear()
            self.pending_times.clear()
            return new_spk_id
        except Exception:
            return None

    def _next_speaker_id(self) -> Optional[int]:
        for idx in range(self.max_speakers):
            if idx not in self.active_spks:
                return idx
        return None


class DiarizationService:
    """Processes audio chunks to emit diarization segments."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_seconds: float = WINDOW_SECONDS,
        stride_seconds: float = STRIDE_SECONDS,
        device: str = "cpu",
        max_speakers: int = MAX_SPEAKERS,
    ) -> None:
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
        self.segment_callback = callback

    def set_active_speaker_callback(self, callback: Callable[[str], None]) -> None:
        self.active_speaker_callback = callback

    def start(self) -> None:
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
        return f"Speaker {speaker_index + 1}"

    def _emit_segment(self, segment: DiarizationSegment) -> None:
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
