"""
Audio processing utilities for waveform visualization and management.

This module provides utilities for managing and visualizing audio waveforms
in real-time transcription applications.
"""

from collections import deque
from typing import List, Optional

import numpy as np


class RollingWaveform:
    """
    Manages a rolling window of audio waveform data for visualization.

    This class maintains a fixed-size buffer of audio waveform points by downsampling
    incoming audio chunks. It's designed for real-time visualization of audio streams
    where you want to display the most recent N seconds of audio data.

    The waveform is represented as normalized amplitude values over time, with automatic
    downsampling to maintain a fixed number of points regardless of the audio length.

    Attributes:
        points_per_second: Number of waveform points to generate per second of audio
        window_seconds: Duration of the rolling window in seconds
        max_points: Maximum number of points to store (points_per_second * window_seconds)
        points: Deque containing the normalized waveform amplitude values
        samples_per_point: Number of audio samples to aggregate into one point
        _remainder: Buffer for audio samples that don't fit into a complete point

    Example:
        >>> waveform = RollingWaveform(points_per_second=20, window_seconds=60)
        >>> # Add audio chunks as they arrive
        >>> waveform.append_chunk(audio_array, sample_rate=16000)
        >>> # Get normalized points for visualization
        >>> points = waveform.get_points()
        >>> print(f"Displaying {len(points)} points for {waveform.window_duration():.1f}s")
    """

    def __init__(
        self,
        points_per_second: int = 20,
        window_seconds: int = 120,
    ) -> None:
        """
        Initialize the RollingWaveform.

        Args:
            points_per_second: Number of waveform points per second (default: 20)
            window_seconds: Duration of the rolling window in seconds (default: 120)
        """
        self.points_per_second = points_per_second
        self.window_seconds = window_seconds
        self.max_points = max(1, points_per_second * window_seconds)
        self.points: deque[float] = deque(maxlen=self.max_points)
        self.samples_per_point: Optional[int] = None
        self._remainder = np.zeros(0, dtype=np.float32)

    def append_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Append a new audio chunk to the waveform.

        This method processes incoming audio data by:
        1. Determining how many audio samples should be aggregated into each point
        2. Combining with any remainder from the previous chunk
        3. Computing the maximum amplitude for each point
        4. Storing any remaining samples that don't form a complete point

        Args:
            audio: Audio data as a numpy array of float32 values
            sample_rate: Sample rate of the audio in Hz

        Example:
            >>> waveform = RollingWaveform(points_per_second=20)
            >>> audio_chunk = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
            >>> waveform.append_chunk(audio_chunk, sample_rate=16000)
        """
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
        """
        Get normalized waveform points for visualization.

        Returns a list of normalized amplitude values where the maximum value
        is scaled to 1.0. If there are no points or the maximum is 0, returns
        a list of zeros.

        Returns:
            List of normalized amplitude values in range [0.0, 1.0]

        Example:
            >>> waveform = RollingWaveform()
            >>> # After adding audio data...
            >>> points = waveform.get_points()
            >>> max_amplitude = max(points) if points else 0
            >>> print(f"Max normalized amplitude: {max_amplitude}")
            Max normalized amplitude: 1.0
        """
        if not self.points:
            return []
        max_val = max(self.points)
        if max_val <= 0.0:
            return [0.0 for _ in self.points]
        return [value / max_val for value in self.points]

    def window_duration(self) -> float:
        """
        Get the current duration of audio data in the window.

        Returns:
            Duration in seconds of the audio data currently stored

        Example:
            >>> waveform = RollingWaveform(points_per_second=20, window_seconds=120)
            >>> # After adding 30 seconds of audio...
            >>> print(f"Window contains {waveform.window_duration():.1f} seconds")
            Window contains 30.0 seconds
        """
        if not self.points:
            return 0.0
        return min(len(self.points), self.max_points) / self.points_per_second

    def reset(self) -> None:
        """
        Clear all waveform data and reset to initial state.

        This method removes all stored points and resets internal buffers,
        useful when starting a new recording session.

        Example:
            >>> waveform = RollingWaveform()
            >>> # After recording...
            >>> waveform.reset()  # Clear for new recording
            >>> print(f"Points after reset: {len(waveform.get_points())}")
            Points after reset: 0
        """
        self.points.clear()
        self.samples_per_point = None
        self._remainder = np.zeros(0, dtype=np.float32)
