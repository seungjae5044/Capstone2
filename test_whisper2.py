import argparse
import copy
import json
import queue
import sys
import threading
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_CONFIG = {
    "model_id": "openai/whisper-large-v3-turbo",
    "language": "ko",
    "force_device": None,
    "audio_source_sr": 48000,
    "target_sr": 16000,
    "chunk_length_s": 12,
    "stride_seconds": 0.4,
    "chunk_duration": 5.0,
    "blocksize_seconds": 0.2,
    "queue_maxsize": 100,
    "batch_size": 1,
    "silence_rms_threshold": 0.005,
    "generate_kwargs": {
        "task": "transcribe",
        "temperature": 0.0,
        "no_speech_threshold": 0.8,
        "logprob_threshold": -1.0,
        "repetition_penalty": 1.05
    }
}


pipe = None
device = "cpu"
torch_dtype = torch.float32

audio_source_sr = DEFAULT_CONFIG["audio_source_sr"]
target_sr = DEFAULT_CONFIG["target_sr"]
chunk_length_s = DEFAULT_CONFIG["chunk_length_s"]
stride_seconds = DEFAULT_CONFIG["stride_seconds"]
chunk_duration = DEFAULT_CONFIG["chunk_duration"]
blocksize = int(audio_source_sr * DEFAULT_CONFIG["blocksize_seconds"])
queue_maxsize = DEFAULT_CONFIG["queue_maxsize"]
batch_size = DEFAULT_CONFIG["batch_size"]
silence_rms_threshold = DEFAULT_CONFIG["silence_rms_threshold"]
resample_up = target_sr // gcd(audio_source_sr, target_sr)
resample_down = audio_source_sr // gcd(audio_source_sr, target_sr)

audio_queue: queue.Queue | None = None
stop_event = threading.Event()


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> dict:
    config_path = Path(path)
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config = deep_update(config, user_config)
    elif path:
        print(f"⚠️ Config file '{path}' not found. Using defaults.", file=sys.stderr)
    return config


def select_device(force_device: str | None) -> tuple[str, torch.dtype]:
    if force_device:
        requested = force_device.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda", torch.float16
        if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        if requested == "cpu":
            return "cpu", torch.float32
        print(f"⚠️ Requested device '{force_device}' is unavailable. Falling back to auto selection.", file=sys.stderr)
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def initialize_from_config(config: dict) -> None:
    global pipe, device, torch_dtype
    global audio_source_sr, target_sr, chunk_length_s, stride_seconds
    global chunk_duration, blocksize, queue_maxsize, batch_size
    global silence_rms_threshold, resample_up, resample_down
    global audio_queue, stop_event

    device, torch_dtype = select_device(config.get("force_device"))

    model_id = config.get("model_id", DEFAULT_CONFIG["model_id"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    chunk_length_s = config.get("chunk_length_s", DEFAULT_CONFIG["chunk_length_s"])
    stride_seconds = config.get("stride_seconds", DEFAULT_CONFIG["stride_seconds"])

    generate_kwargs = copy.deepcopy(DEFAULT_CONFIG["generate_kwargs"])
    generate_kwargs = deep_update(generate_kwargs, config.get("generate_kwargs", {}))
    language = config.get("language")
    if language:
        generate_kwargs.setdefault("language", language)

    pipe_kwargs = {
        "model": model,
        "tokenizer": processor.tokenizer,
        "feature_extractor": processor.feature_extractor,
        "torch_dtype": torch_dtype,
        "device": device,
        "chunk_length_s": chunk_length_s,
        "stride_length_s": (stride_seconds, stride_seconds),
        "return_timestamps": True,
        "generate_kwargs": generate_kwargs,
    }
    pipe_instance = pipeline("automatic-speech-recognition", **pipe_kwargs)
    pipe = pipe_instance

    audio_source_sr = config.get("audio_source_sr", DEFAULT_CONFIG["audio_source_sr"])
    target_sr = config.get("target_sr", DEFAULT_CONFIG["target_sr"])
    chunk_duration = config.get("chunk_duration", DEFAULT_CONFIG["chunk_duration"])
    blocksize_seconds = config.get("blocksize_seconds", DEFAULT_CONFIG["blocksize_seconds"])
    blocksize = max(1, int(audio_source_sr * blocksize_seconds))
    queue_maxsize = config.get("queue_maxsize", DEFAULT_CONFIG["queue_maxsize"])
    batch_size = config.get("batch_size", DEFAULT_CONFIG["batch_size"])
    silence_rms_threshold = config.get("silence_rms_threshold", DEFAULT_CONFIG["silence_rms_threshold"])

    factor = gcd(audio_source_sr, target_sr)
    resample_up = target_sr // factor
    resample_down = audio_source_sr // factor

    audio_queue = queue.Queue(maxsize=queue_maxsize)
    stop_event = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Whisper transcription")
    parser.add_argument(
        "--config",
        type=str,
        default="config_whisper.json",
        help="Path to JSON configuration file",
    )
    return parser.parse_args()


def callback(indata, frames, time_info, status):
    if status:
        print(f"SoundDevice Status: {status}", file=sys.stderr)
    if audio_queue is None:
        return
    audio_data = indata.copy().astype(np.float32)
    try:
        audio_queue.put_nowait(audio_data)
    except queue.Full:
        pass


def transcribe_stream() -> None:
    if pipe is None or audio_queue is None:
        raise RuntimeError("Pipeline is not initialized")

    full_audio_buffer: list[np.ndarray] = []
    prev_transcription = ""

    with sd.InputStream(
        samplerate=audio_source_sr,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        print("\n👂 마이크 녹음 및 실시간 전사 시작! (Ctrl+C를 눌러 종료하세요)")

        try:
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=chunk_duration)
                except queue.Empty:
                    continue

                if chunk.ndim > 1:
                    chunk = chunk[:, 0]

                resampled_chunk = resample_poly(chunk, up=resample_up, down=resample_down)
                full_audio_buffer.append(resampled_chunk)

                total_samples = sum(len(buffer_chunk) for buffer_chunk in full_audio_buffer)
                total_duration = total_samples / target_sr

                if total_duration >= chunk_duration:
                    audio_data_np = np.concatenate(full_audio_buffer)
                    rms_level = np.sqrt(np.mean(np.square(audio_data_np))) if audio_data_np.size else 0.0
                    if rms_level < silence_rms_threshold:
                        full_audio_buffer = []
                        continue

                    with torch.inference_mode():
                        result = pipe(
                            audio_data_np,
                            batch_size=batch_size,
                        )

                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    current_text = result.get("text", "").strip()
                    prev_text = prev_transcription.strip()

                    if current_text.startswith(prev_text) and len(current_text) > len(prev_text):
                        new_segment = current_text[len(prev_text):].strip()
                    else:
                        new_segment = current_text

                    if new_segment:
                        print(f"💬 {new_segment}", end=" ", flush=True)

                    prev_transcription = current_text

                    overlap_samples = int(target_sr * stride_seconds)
                    if overlap_samples > 0 and audio_data_np.size > overlap_samples:
                        full_audio_buffer = [audio_data_np[-overlap_samples:]]
                    else:
                        full_audio_buffer = [audio_data_np]

        except KeyboardInterrupt:
            print("\n🛑 사용자 요청으로 실시간 전사를 종료합니다.")
            stop_event.set()
        except Exception as exc:
            print(f"\n🚨 오류 발생: {exc}")
            stop_event.set()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    initialize_from_config(config)
    transcribe_stream()


if __name__ == "__main__":
    main()
