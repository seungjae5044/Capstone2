# src_B - Refactored Backend

Modular backend for real-time meeting transcription, speaker diarization, and AI-powered analysis.

## 📁 Directory Structure

```
src_B/
├── __init__.py                 # Package initialization
├── main.py                     # Main entry point (API or Terminal mode)
├── terminal_interface.py       # Terminal CLI interface
│
├── models/                     # Data models and ML models
│   ├── __init__.py
│   ├── data_models.py         # All dataclasses (WhisperConfig, TranscribedSegment, etc.)
│   └── ml_models.py           # VAD and Encoder (SileroVAD, SpeechBrainEncoder)
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── config.py              # Config loading, device selection
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── audio.py               # RollingWaveform for audio visualization
│   └── timeline.py            # TimelineManager for speaker timeline
│
├── diarization/                # Speaker diarization service
│   ├── __init__.py
│   ├── constants.py           # Diarization constants
│   ├── speaker_handler.py     # DiarizationSpeakerHandler
│   └── service.py             # DiarizationService (main service)
│
├── transcription/              # Speech transcription service
│   ├── __init__.py
│   ├── speaker_handler.py     # TranscriptionSpeakerHandler
│   └── service.py             # TranscriptionService (Whisper-based)
│
├── evaluation/                 # AI-powered evaluation
│   ├── __init__.py
│   ├── evaluator.py           # OllamaEvaluator (Gemma-based)
│   └── helpers.py             # Helper functions (JSON extraction)
│
├── reports/                    # Report generation
│   ├── __init__.py
│   ├── constants.py           # Report constants (REPORTS_DIR)
│   ├── summarizer.py          # Summary generation (Gemma)
│   ├── pdf_generator.py       # PDF report generation
│   └── manager.py             # Report orchestration
│
└── api/                        # FastAPI web service
    ├── __init__.py
    ├── app.py                 # FastAPI application
    ├── routes.py              # API endpoints
    ├── websocket.py           # WebSocket management
    └── state.py               # Global state management
```

## 🚀 Usage

### Running the API Server

```bash
# Default mode (API server on http://0.0.0.0:8000)
python -m src_B.main

# Custom host and port
python -m src_B.main --host 127.0.0.1 --port 8080

# Development mode with auto-reload
python -m src_B.main --reload
```

### Running in Terminal Mode

```bash
# Interactive terminal mode
python -m src_B.main --terminal

# With predefined topic and speakers
python -m src_B.main --terminal --topic "Sprint Planning" --speakers 4

# Force specific device
python -m src_B.main --terminal --device cuda

# With custom config
python -m src_B.main --terminal --config config_whisper.json
```

### API Endpoints

- `GET /health` - Health check
- `GET /api/status` - Current meeting status
- `GET /api/speakers` - Speaker summaries and statistics
- `GET /api/timeline` - Timeline segments
- `POST /api/start` - Start meeting session
- `POST /api/stop/{session_id}` - Stop meeting session
- `GET /api/report?format=pdf|json` - Download report
- `WebSocket /ws` - Real-time updates

## 📦 Key Components

### 1. **Transcription Service** (`transcription/service.py`)
- Real-time speech-to-text using Whisper
- Integrated VAD (Voice Activity Detection)
- Speaker identification and tracking
- Audio visualization support

### 2. **Diarization Service** (`diarization/service.py`)
- Real-time speaker diarization
- Speaker embedding extraction
- Automatic speaker clustering
- VAD-based speech detection

### 3. **Evaluation Service** (`evaluation/evaluator.py`)
- AI-powered statement evaluation using Ollama/Gemma
- Topic relevance scoring (0-10)
- Novelty/idea freshness scoring (0-10)
- Meeting statistics tracking

### 4. **Report Generation** (`reports/`)
- AI-generated meeting summaries
- Professional PDF reports
- Key discussion points extraction
- Speaker analysis and metrics

## 🔧 Configuration

### Whisper Configuration

Default config can be overridden with `config_whisper.json`:

```json
{
  "model_id": "openai/whisper-large-v3-turbo",
  "language": "ko",
  "audio_source_sr": 48000,
  "target_sr": 16000,
  "chunk_length_s": 12,
  "chunk_duration": 5.0,
  "batch_size": 1
}
```

### Environment Variables

```bash
# Ollama configuration
export OLLAMA_MODEL="gemma2:9b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

## 🛠️ Development

### Module Dependencies

```python
# Core data models
from src_B.models.data_models import (
    WhisperConfig,
    TranscribedSegment,
    SpeakerProfile,
    TimelineSegment,
    DiarizationSegment,
    MeetingState,
    MeetingStatistics,
)

# ML models
from src_B.models.ml_models import SileroVAD, SpeechBrainEncoder

# Services
from src_B.transcription.service import TranscriptionService
from src_B.diarization.service import DiarizationService
from src_B.evaluation.evaluator import OllamaEvaluator

# Reports
from src_B.reports import finalize_meeting_report
```

### Running Tests

```bash
# Test imports
python -c "import src_B; print('Import successful')"

# Test terminal interface help
python -m src_B.main --help

# Test API server (Ctrl+C to stop)
python -m src_B.main
```

## 🔄 Migration from Old Code

### Old Structure
```
whisper_web_ui.py           (2314 lines)
diarization_service.py      (438 lines)
```

### New Structure
```
src_B/                      (Modular, ~7000+ lines across 30+ files)
```

### Benefits
- ✅ Modular, maintainable code structure
- ✅ Clear separation of concerns
- ✅ Reusable components
- ✅ Better testability
- ✅ Comprehensive documentation
- ✅ Terminal interface (Web UI removed)
- ✅ API-only architecture

## 📝 Notes

1. **Web UI Removed**: The old HTML dashboard has been removed. Use the frontend (src_F) or terminal interface instead.

2. **Terminal Interface**: New CLI interface for running transcription without a web browser.

3. **Thread Safety**: All services use proper locking and thread-safe operations.

4. **Error Handling**: Comprehensive error handling with logging throughout.

5. **Optional Dependencies**: ReportLab is optional for PDF generation.

## 🐛 Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install fastapi uvicorn torch transformers sounddevice scipy scikit-learn speechbrain requests
pip install reportlab  # Optional, for PDF reports
```

### CUDA/MPS Issues
Force CPU mode if GPU issues occur:
```bash
python -m src_B.main --terminal --device cpu
```

### Ollama Not Available
Evaluation will fail gracefully if Ollama is not running. Start Ollama:
```bash
ollama serve
```

## 📄 License

Copyright © 2025 Capstone2 Team
