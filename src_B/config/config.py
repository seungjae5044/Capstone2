"""
Configuration management for Whisper transcription service.

This module provides configuration loading, merging, and device selection utilities
for the Whisper-based transcription system.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# Configuration file path
CONFIG_PATH = Path("config_whisper.json")


# Default configuration for Whisper transcription
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_id": "openai/whisper-large-v3-turbo",
    "language": "ko",
    "force_device": None,
    "audio_source_sr": 48000,
    "target_sr": 16000,
    "chunk_length_s": 5,
    "stride_seconds": 0.8,
    "chunk_duration": 5.0,
    "blocksize_seconds": 0.2,
    "queue_maxsize": 100,
    "batch_size": 1,
    "silence_rms_threshold": 0.02,
    "generate_kwargs": {
        "task": "transcribe",
        "temperature": 0.0,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "repetition_penalty": 1.2,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    This function performs a deep merge of nested dictionaries. For nested dict values,
    it recursively merges them. For other values, the update value overwrites the base value.

    Args:
        base: The base dictionary to be updated
        updates: The dictionary containing updates to apply

    Returns:
        The merged dictionary (note: base is modified in-place)

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> updates = {"b": {"d": 4, "e": 5}, "f": 6}
        >>> result = deep_update(base, updates)
        >>> result
        {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration from a JSON file and merge with defaults.

    This function loads a configuration file from the specified path and merges it
    with the DEFAULT_CONFIG. If the file doesn't exist, it returns a copy of the
    DEFAULT_CONFIG.

    Args:
        path: Path to the configuration JSON file

    Returns:
        A dictionary containing the merged configuration

    Example:
        >>> config = load_config(Path("config_whisper.json"))
        >>> config["model_id"]
        'openai/whisper-large-v3-turbo'
    """
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config = deep_update(config, user_config)
    return config


def select_device(force_device: Optional[str]) -> Tuple[str, torch.dtype]:
    """
    Select the best available device for PyTorch computation.

    This function determines the appropriate device (CUDA, MPS, or CPU) and data type
    for PyTorch operations. If a specific device is requested via force_device, it
    attempts to use that device if available. Otherwise, it automatically selects
    the best available device.

    Device priority (when auto-selecting):
    1. CUDA (NVIDIA GPU) with float16
    2. MPS (Apple Silicon) with float16
    3. CPU with float32

    Args:
        force_device: Optional device name to force ("cuda", "mps", or "cpu").
                     If None or unavailable, automatically selects best device.

    Returns:
        A tuple of (device_name, torch_dtype) where:
        - device_name is "cuda", "mps", or "cpu"
        - torch_dtype is torch.float16 for GPU devices, torch.float32 for CPU

    Example:
        >>> device, dtype = select_device(None)
        >>> print(f"Using {device} with {dtype}")
        Using cuda with torch.float16

        >>> device, dtype = select_device("cpu")
        >>> print(f"Using {device} with {dtype}")
        Using cpu with torch.float32
    """
    if force_device:
        requested = force_device.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda", torch.float16
        if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16
        if requested == "cpu":
            return "cpu", torch.float32
        logger.warning("요청한 장치 %s 를 사용할 수 없습니다. 자동 선택으로 진행합니다.", force_device)

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32
