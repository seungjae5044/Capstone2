"""
Timeline management for speaker segments in meetings.

This module provides utilities for tracking and managing speaker timeline segments
during meeting transcription sessions, including speaker tracking and timeline visualization.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src_B.models.data_models import TimelineSegment


class TimelineManager:
    """
    Manages a timeline of speaker segments during a meeting.

    This class maintains a thread-safe list of speaker segments, automatically merging
    consecutive segments from the same speaker. It's designed for real-time meeting
    transcription where speaker changes need to be tracked over time.

    The timeline is represented as a sequence of segments, where each segment contains:
    - Speaker identification (ID and name)
    - Start and end timestamps
    - Duration information

    Attributes:
        _segments: Internal list of TimelineSegment objects
        _lock: Threading lock for thread-safe operations

    Example:
        >>> timeline = TimelineManager()
        >>> start = datetime.now()
        >>> end = datetime.now()
        >>> timeline.add_segment("speaker_1", "Alice", start, end)
        >>> segments = timeline.snapshot()
        >>> print(f"Timeline has {len(segments)} segments")
    """

    def __init__(self) -> None:
        """
        Initialize the TimelineManager with an empty timeline.
        """
        self._segments: List[TimelineSegment] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        """
        Clear all timeline segments.

        This method removes all stored segments, useful when starting a new
        meeting session.

        Example:
            >>> timeline = TimelineManager()
            >>> # After recording a meeting...
            >>> timeline.reset()  # Clear for new meeting
            >>> print(f"Segments after reset: {len(timeline.snapshot())}")
            Segments after reset: 0
        """
        with self._lock:
            self._segments = []

    def add_segment(self, speaker_id: str, speaker_name: str, start: datetime, end: datetime) -> None:
        """
        Add a new speaker segment to the timeline.

        This method adds a speaker segment and automatically merges it with the
        previous segment if they have the same speaker. This prevents fragmentation
        of continuous speech from the same speaker.

        Args:
            speaker_id: Unique identifier for the speaker (e.g., "speaker_1")
            speaker_name: Display name for the speaker (e.g., "Alice")
            start: Start timestamp of the segment
            end: End timestamp of the segment

        Example:
            >>> timeline = TimelineManager()
            >>> from datetime import datetime, timedelta
            >>> now = datetime.now()
            >>>
            >>> # Add first segment
            >>> timeline.add_segment("speaker_1", "Alice", now, now + timedelta(seconds=5))
            >>>
            >>> # Add consecutive segment from same speaker (will be merged)
            >>> timeline.add_segment("speaker_1", "Alice",
            ...                     now + timedelta(seconds=5),
            ...                     now + timedelta(seconds=10))
            >>>
            >>> # Add segment from different speaker (new segment)
            >>> timeline.add_segment("speaker_2", "Bob",
            ...                     now + timedelta(seconds=10),
            ...                     now + timedelta(seconds=15))
            >>>
            >>> print(f"Timeline has {len(timeline.snapshot())} segments")
            Timeline has 2 segments
        """
        with self._lock:
            if self._segments and self._segments[-1].speaker_id == speaker_id:
                self._segments[-1].end_time = end
            else:
                self._segments.append(
                    TimelineSegment(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        start_time=start,
                        end_time=end,
                    )
                )

    def snapshot(self) -> List[Dict[str, Any]]:
        """
        Get a snapshot of all timeline segments.

        Returns a thread-safe copy of all segments as dictionaries, suitable
        for serialization or display.

        Returns:
            List of dictionaries, each representing a segment with keys:
            - speaker_id: Speaker identifier
            - speaker_name: Speaker display name
            - start_time: Segment start timestamp (ISO format)
            - end_time: Segment end timestamp (ISO format)
            - duration: Segment duration in seconds

        Example:
            >>> timeline = TimelineManager()
            >>> # After adding segments...
            >>> segments = timeline.snapshot()
            >>> for seg in segments:
            ...     print(f"{seg['speaker_name']}: {seg['duration']:.1f}s")
            Alice: 10.0s
            Bob: 5.0s
        """
        with self._lock:
            return [segment.to_dict() for segment in self._segments]

    def latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent timeline segment.

        Returns:
            Dictionary representing the latest segment, or None if timeline is empty.
            The dictionary has the same structure as returned by snapshot().

        Example:
            >>> timeline = TimelineManager()
            >>> # After adding segments...
            >>> latest = timeline.latest()
            >>> if latest:
            ...     print(f"Currently speaking: {latest['speaker_name']}")
            Currently speaking: Bob
        """
        with self._lock:
            if not self._segments:
                return None
            return self._segments[-1].to_dict()
