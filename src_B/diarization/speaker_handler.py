"""Speaker handler for maintaining speaker embeddings and classification.

This module provides the SpeakerHandler class which manages speaker embeddings,
performs speaker classification, and handles pending speaker promotion.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src_B.diarization.constants import (
    AUTO_CLUSTER_DISTANCE_THRESHOLD,
    EMBEDDING_UPDATE_THRESHOLD,
    MAX_SPEAKERS,
    MIN_CLUSTER_SIZE,
    MIN_PENDING_SIZE,
    PENDING_THRESHOLD,
)
from src_B.models.data_models import SpeakerDecision


class SpeakerHandler:
    """Maintains running speaker embeddings and pending queue.

    This class manages speaker identification by maintaining embeddings for known
    speakers and a pending queue for potential new speakers. It performs clustering
    on pending embeddings to automatically detect new speakers.

    Attributes:
        max_speakers: Maximum number of concurrent speakers to track
        change_threshold: Similarity threshold for speaker change detection
        min_pending: Minimum pending samples before attempting promotion
        mean_embs: Mean embeddings for each speaker slot
        spk_embs: All embeddings for each speaker
        active_spks: Set of currently active speaker indices
        pending_embs: Queue of embeddings not yet assigned to a speaker
        pending_times: Timestamps corresponding to pending embeddings
    """

    def __init__(
        self,
        max_speakers: int = MAX_SPEAKERS,
        change_threshold: float = PENDING_THRESHOLD,
        min_pending: int = MIN_PENDING_SIZE,
    ) -> None:
        """Initialize the speaker handler.

        Args:
            max_speakers: Maximum number of speakers to track (default: MAX_SPEAKERS)
            change_threshold: Similarity threshold for speaker changes (default: PENDING_THRESHOLD)
            min_pending: Minimum pending samples before promotion (default: MIN_PENDING_SIZE)
        """
        self.max_speakers = max_speakers
        self.change_threshold = change_threshold
        self.min_pending = min_pending
        self.mean_embs: List[Optional[np.ndarray]] = [None] * max_speakers
        self.spk_embs: List[List[np.ndarray]] = [[] for _ in range(max_speakers)]
        self.active_spks: set[int] = set()
        self.pending_embs: List[np.ndarray] = []
        self.pending_times: List[float] = []

    def reset(self) -> None:
        """Reset all speaker data and pending queue."""
        self.mean_embs = [None] * self.max_speakers
        self.spk_embs = [[] for _ in range(self.max_speakers)]
        self.active_spks.clear()
        self.pending_embs.clear()
        self.pending_times.clear()

    def classify(self, emb: np.ndarray, seg_start_time: float) -> SpeakerDecision:
        """Classify an embedding to a speaker or mark as pending.

        This method compares the input embedding against known speaker embeddings
        and either assigns it to an existing speaker or adds it to the pending queue.

        Args:
            emb: Speaker embedding vector to classify
            seg_start_time: Start time of the audio segment for this embedding

        Returns:
            SpeakerDecision object containing:
                - speaker_index: Index of matched speaker or None if pending
                - similarity: Cosine similarity to matched speaker
                - is_pending: Whether this embedding is in pending queue
                - promoted_speaker_index: If a pending speaker was promoted, its index
        """
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
        """Attempt to promote pending embeddings to a new speaker.

        Uses agglomerative clustering to identify coherent clusters in the pending
        queue. If a cluster is large enough, it's promoted to a new speaker.

        Returns:
            Index of the newly promoted speaker, or None if no promotion occurred
        """
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
        """Find the next available speaker slot index.

        Returns:
            First unused speaker index, or None if all slots are occupied
        """
        for idx in range(self.max_speakers):
            if idx not in self.active_spks:
                return idx
        return None
