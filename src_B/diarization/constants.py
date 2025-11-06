"""Constants for diarization service.

This module contains all configuration constants used by the diarization system.
"""

# Audio processing parameters
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
STRIDE_SECONDS = 0.1

# VAD and speaker classification thresholds
VAD_THRESHOLD = 0.5
PENDING_THRESHOLD = 0.3
EMBEDDING_UPDATE_THRESHOLD = 0.4

# Speaker management parameters
MIN_PENDING_SIZE = 30
AUTO_CLUSTER_DISTANCE_THRESHOLD = 0.6
MIN_CLUSTER_SIZE = 15
MAX_SPEAKERS = 10
