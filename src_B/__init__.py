"""
Meeting Transcription System with Speaker Diarization.

A refactored, modular backend for real-time meeting transcription,
speaker diarization, and AI-powered analysis.

Modules:
    - models: Data models and ML models (VAD, Encoder)
    - config: Configuration management
    - utils: Utility functions (audio, timeline)
    - diarization: Speaker diarization service
    - transcription: Speech transcription service
    - evaluation: AI-powered evaluation (Ollama)
    - reports: Report generation (PDF, summary)
    - api: FastAPI routes and WebSocket
    - terminal_interface: CLI interface
    - main: Main entry point

Usage:
    # Run API server
    python -m src_B.main

    # Run terminal interface
    python -m src_B.main --terminal
"""

__version__ = "2.0.0"
__author__ = "Capstone2 Team"
