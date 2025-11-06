"""
Speaker identification and tracking for real-time transcription.

This module provides the SpeakerHandler class for managing speaker profiles
and classifying speech segments based on speaker embeddings. It uses similarity
matching and exponential moving averages to maintain stable speaker identities.

Note: This is the transcription-side speaker handler, which is different from
the diarization-side speaker handler.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src_B.models.data_models import SpeakerProfile


class SpeakerHandler:
    """
    Manages speaker identification and profile tracking for transcription.

    This class maintains a collection of speaker profiles and classifies new audio
    segments by comparing speaker embeddings. It uses cosine similarity for matching
    and exponential moving average for updating speaker embeddings over time.

    Key features:
    - Thread-safe operations for concurrent access
    - Automatic speaker profile creation when similarity is below threshold
    - Profile merging through embedding updates
    - Maximum speaker limit to prevent unbounded growth

    Attributes:
        max_speakers: Maximum number of unique speakers to track
        similarity_threshold: Minimum cosine similarity to match existing speaker
        update_alpha: Learning rate for exponential moving average of embeddings
        _counter: Internal counter for assigning sequential speaker IDs
        _profiles: Dictionary mapping speaker_id to SpeakerProfile
        _lock: Threading lock for thread-safe operations

    Example:
        >>> handler = SpeakerHandler(max_speakers=10, similarity_threshold=0.35)
        >>> embedding = np.random.rand(192)  # Example embedding
        >>> speaker_id, profile, similarity = handler.classify(embedding)
        >>> handler.register_segment(speaker_id, duration=5.0, similarity=similarity)
        >>> profiles = handler.get_profiles()
        >>> print(f"Tracking {len(profiles)} speakers")
    """

    def __init__(
        self,
        max_speakers: int = 10,
        similarity_threshold: float = 0.35,
        update_alpha: float = 0.2,
    ) -> None:
        """
        Initialize the SpeakerHandler.

        Args:
            max_speakers: Maximum number of unique speakers to track (default: 10)
            similarity_threshold: Minimum cosine similarity for speaker matching (default: 0.35)
                               Lower values are more lenient, higher values require closer matches
            update_alpha: Learning rate for embedding updates (default: 0.2)
                        Higher values adapt faster to new embeddings, lower values are more stable
        """
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.update_alpha = update_alpha
        self._counter = 1
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """
        Reset all speaker profiles and counters.

        This method clears all tracked speakers and resets the speaker counter.
        Useful when starting a new meeting or session.

        Example:
            >>> handler = SpeakerHandler()
            >>> # After a meeting...
            >>> handler.reset()  # Clear for new meeting
            >>> print(f"Profiles after reset: {len(handler.get_profiles())}")
            Profiles after reset: 0
        """
        with self._lock:
            self._counter = 1
            self._profiles.clear()

    def classify(self, embedding: Optional[np.ndarray]) -> Tuple[str, SpeakerProfile, float]:
        """
        Classify a speaker based on their voice embedding.

        This method compares the provided embedding against all known speaker profiles
        using cosine similarity. If the best match is above the similarity threshold,
        the embedding is assigned to that speaker and the profile is updated. Otherwise,
        a new speaker profile is created.

        Args:
            embedding: Speaker embedding vector as numpy array, or None for default speaker

        Returns:
            A tuple of (speaker_id, profile, similarity) where:
            - speaker_id: Unique identifier for the speaker (e.g., "speaker_1")
            - profile: SpeakerProfile object containing speaker information
            - similarity: Cosine similarity score (0.0 to 1.0), or 0.0 for default assignment

        Example:
            >>> handler = SpeakerHandler()
            >>> embedding = np.random.rand(192)
            >>> speaker_id, profile, similarity = handler.classify(embedding)
            >>> print(f"Assigned to {speaker_id} with similarity {similarity:.2f}")
            Assigned to speaker_1 with similarity 1.00
        """
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
        """
        Register a transcribed segment for a speaker.

        This method updates the speaker's statistics including statement count,
        total duration, and last similarity score.

        Args:
            speaker_id: Unique identifier for the speaker
            duration: Duration of the segment in seconds
            similarity: Similarity score from classification

        Example:
            >>> handler = SpeakerHandler()
            >>> # After classifying a segment...
            >>> handler.register_segment("speaker_1", duration=5.0, similarity=0.85)
            >>> profile = handler.get_profile("speaker_1")
            >>> print(f"Total duration: {profile.duration:.1f}s")
        """
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
        """
        Assign speech to a default speaker when embedding is unavailable.

        If no profiles exist, creates a new profile. Otherwise assigns to the
        first existing speaker.

        Returns:
            A tuple of (speaker_id, profile, 0.0)

        Note:
            This is an internal method. Similarity is always 0.0 for default assignments.
        """
        with self._lock:
            if not self._profiles:
                speaker_id, profile = self._create_profile(np.zeros(1, dtype=np.float32))
            else:
                speaker_id = next(iter(self._profiles))
                profile = self._profiles[speaker_id]
            return speaker_id, profile, 0.0

    def _create_profile(self, embedding: np.ndarray) -> Tuple[str, SpeakerProfile]:
        """
        Create a new speaker profile.

        This method generates a new speaker ID, normalizes the embedding,
        and creates a SpeakerProfile instance.

        Args:
            embedding: Speaker embedding vector to initialize the profile

        Returns:
            A tuple of (speaker_id, profile)

        Note:
            This is an internal method. The counter is capped at max_speakers + 1
            to prevent overflow while allowing max_speakers unique profiles.
        """
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
        """
        Get all tracked speaker profiles.

        Returns:
            List of all SpeakerProfile objects in thread-safe manner

        Example:
            >>> handler = SpeakerHandler()
            >>> # After some classification...
            >>> profiles = handler.get_profiles()
            >>> for profile in profiles:
            ...     print(f"{profile.display_name}: {profile.statement_count} statements")
            Speaker 1: 5 statements
            Speaker 2: 3 statements
        """
        with self._lock:
            return [profile for profile in self._profiles.values()]

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """
        Get a specific speaker profile by ID.

        Args:
            speaker_id: Unique identifier for the speaker

        Returns:
            SpeakerProfile object if found, None otherwise

        Example:
            >>> handler = SpeakerHandler()
            >>> profile = handler.get_profile("speaker_1")
            >>> if profile:
            ...     print(f"Found profile: {profile.display_name}")
            Found profile: Speaker 1
        """
        with self._lock:
            return self._profiles.get(speaker_id)
