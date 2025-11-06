"""ML models for speech processing: VAD and speaker embedding extraction."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch


class SileroVAD:
    """Lightweight wrapper around Silero VAD with thread-safe lazy initialization."""

    def __init__(self, threshold: float = 0.5, device: str = "cpu") -> None:
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech detection threshold (0.0-1.0)
            device: Device to run model on ('cpu', 'cuda')
        """
        self.threshold = threshold
        self.device = device
        self.model = None
        self.get_speech_ts = None
        self.lock = threading.Lock()

    def _initialize(self) -> None:
        """Load Silero VAD model (thread-safe, lazy initialization)."""
        with self.lock:
            if self.model is not None and self.get_speech_ts is not None:
                return
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.model = model.to("cpu")  # Silero VAD works best on CPU
            self.get_speech_ts = utils[0]

    def is_speech(self, audio: np.ndarray, sr: int) -> bool:
        """
        Detect if audio contains speech.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of audio

        Returns:
            True if speech detected, False otherwise
        """
        # Require minimum audio length
        if audio.size < max(1600, int(sr * 0.2)):
            return False

        # Lazy initialization
        if self.model is None or self.get_speech_ts is None:
            self._initialize()

        if self.model is None or self.get_speech_ts is None:
            return False

        # Convert to tensor and detect speech
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
    """Speaker embedding extraction using SpeechBrain ECAPA-TDNN."""

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize SpeechBrain encoder.

        Args:
            device: Device to run model on ('cpu', 'cuda')
        """
        self.device = device if device == "cuda" else "cpu"
        self.encoder = None
        self.lock = threading.Lock()

    def _initialize(self) -> None:
        """Load SpeechBrain encoder model (thread-safe, lazy initialization)."""
        with self.lock:
            if self.encoder is not None:
                return
            from speechbrain.pretrained import EncoderClassifier

            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )

    def embed(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of audio

        Returns:
            Normalized embedding vector or None if extraction fails
        """
        if audio.size == 0:
            return None

        # Lazy initialization
        if self.encoder is None:
            self._initialize()

        if self.encoder is None:
            return None

        # Convert to tensor and extract embedding
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        waveform = waveform.to(self.device)

        with torch.no_grad():
            embedding = self.encoder.encode_batch(waveform)

        # Normalize embedding
        emb = embedding.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm
