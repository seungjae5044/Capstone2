# API Extraction Summary

This document summarizes the extraction of API, WebSocket, and state management code from `whisper_web_ui.py` into modular components in the `src_B/api/` directory.

## Created Files

### 1. `/home/user/Capstone2/src_B/api/state.py` (327 lines)

**Purpose**: Global state management for the meeting application

**Key Components**:
- **Global Variables**:
  - `meeting_state` - Current meeting session state (MeetingState instance)
  - `meeting_state_lock` - Thread lock for meeting state
  - `meeting_stats` - Meeting statistics tracker (MeetingStatistics instance)
  - `meeting_history` - List of all meeting statements with evaluations
  - `meeting_history_lock` - Thread lock for history
  - `transcription_service` - TranscriptionService instance
  - `diarization_service` - Optional DiarizationService instance
  - `ollama_evaluator` - OllamaEvaluator instance for AI evaluation
  - `last_report_path` - Path to generated PDF report
  - `last_summary_payload` - JSON summary data
  - Deduplication tracking variables

- **Helper Functions**:
  - `set_app_state()` / `get_app_state()` - Manage FastAPI app state reference
  - `enqueue_event()` - Queue messages for WebSocket broadcast
  - `broadcast_session_status()` - Broadcast session status to clients
  - `build_speaker_stats_payload()` - Build comprehensive speaker statistics
  - `broadcast_stats_messages()` - Broadcast statistics updates
  - `handle_transcription()` - Process new transcription segments
  - `schedule_evaluation()` - Schedule AI evaluation tasks
  - `ensure_diarization_service()` - Initialize/manage diarization service
  - `handle_diarization_segment()` - Process diarization segments
  - `update_active_speaker()` - Update currently active speaker

**Dependencies**:
- `src_B.models.data_models` - MeetingState, MeetingStatistics, TranscribedSegment
- `src_B.transcription.service` - TranscriptionService
- `src_B.diarization.service` - DiarizationService, DiarizationSegment, MAX_SPEAKERS
- `src_B.evaluation.evaluator` - OllamaEvaluator

---

### 2. `/home/user/Capstone2/src_B/api/websocket.py` (99 lines)

**Purpose**: WebSocket connection management for real-time updates

**Key Components**:
- **ConnectionManager Class**:
  - `connect()` - Accept and register new WebSocket connections
  - `disconnect()` - Unregister WebSocket connections
  - `broadcast()` - Send messages to all connected clients
  - Thread-safe connection management with asyncio.Lock

- **Global Instance**:
  - `manager` - Singleton ConnectionManager instance

- **Functions**:
  - `broadcast_worker()` - Background task that processes event queue
  - `websocket_endpoint()` - FastAPI WebSocket endpoint handler

**Features**:
- Automatic cleanup of failed connections
- Thread-safe connection list management
- Graceful handling of WebSocket disconnections
- Continuous event broadcasting from queue

---

### 3. `/home/user/Capstone2/src_B/api/routes.py` (313 lines)

**Purpose**: FastAPI REST API route handlers

**Endpoints**:

#### Health & Status
- `GET /health` - Health check endpoint
- `GET /api/status` - Get current meeting status

#### Meeting Control
- `POST /api/start` - Start a new meeting session
  - Parameters: topic (required), speaker_id, expected_speakers
  - Initializes transcription and diarization services
  - Returns: session_id, topic
  
- `POST /api/stop/{session_id}` - Stop the current meeting
  - Stops all services
  - Generates meeting report
  - Broadcasts final state

#### Data Retrieval
- `GET /api/speakers` - Get speaker summaries and statistics
- `GET /api/timeline` - Get timeline of speaker segments
- `GET /api/report` - Get meeting report (PDF or JSON)
  - Query param: format=pdf (default) or format=json

**Error Handling**:
- HTTPException for validation errors
- Graceful rollback on service startup failures
- Session ID validation

**Dependencies**:
- All state management functions from `src_B.api.state`
- `src_B.reports.generator.finalize_meeting_report`
- `src_B.diarization.service.MAX_SPEAKERS`

---

### 4. `/home/user/Capstone2/src_B/api/app.py` (165 lines)

**Purpose**: FastAPI application initialization and lifecycle management

**Key Components**:

- **FastAPI App Creation**:
  - Title: "Whisper Meeting API"
  - Description: "Real-time meeting transcription and evaluation API (No Web UI)"
  - Version: "1.0.0"

- **Background Workers**:
  - `evaluation_worker()` - Processes evaluation tasks from queue
  - `_evaluate_and_broadcast()` - Evaluates transcriptions and broadcasts results
  - `broadcast_worker()` - Broadcasts events to WebSocket clients (from websocket.py)

- **Lifecycle Events**:
  - `on_startup()`:
    - Creates event loop reference
    - Initializes event and evaluation queues
    - Starts background worker tasks
    - Sets app state reference in state module
  
  - `on_shutdown()`:
    - Cancels background worker tasks
    - Graceful cleanup

- **Route Registration**:
  - Includes all routes from `routes.py`
  - WebSocket endpoint at `/ws`

**Architecture**:
- Separation of concerns: routes, state, websocket are separate modules
- Background workers for async processing
- Centralized state management
- Event-driven architecture with asyncio queues

---

## Architecture Overview

```
src_B/api/
├── __init__.py
├── state.py          # Global state + helper functions
├── websocket.py      # WebSocket connection management
├── routes.py         # REST API endpoints
└── app.py           # FastAPI app + lifecycle management
```

### Data Flow

1. **Incoming Requests** → `routes.py` endpoints
2. **State Changes** → `state.py` helper functions
3. **Events** → `enqueue_event()` → event_queue
4. **Broadcasting** → `broadcast_worker()` → WebSocket clients
5. **Transcriptions** → `handle_transcription()` → `schedule_evaluation()`
6. **Evaluations** → `evaluation_worker()` → `_evaluate_and_broadcast()`

### Key Design Decisions

1. **Removed HTML Dashboard**: 
   - Original `get_dashboard_html()` function not included
   - API-only architecture (no web UI)

2. **Modular State Management**:
   - All global state in `state.py`
   - Accessors via functions (set_app_state, get_app_state)

3. **Background Workers**:
   - Event broadcasting in separate async task
   - Evaluation processing in separate async task
   - Non-blocking API endpoints

4. **Error Handling**:
   - Graceful degradation
   - Proper cleanup on failures
   - Detailed error messages in HTTPException

---

## Dependencies Required

The extracted modules depend on the following `src_B` modules:

- `src_B.models.data_models`:
  - MeetingState
  - MeetingStatistics
  - TranscribedSegment

- `src_B.transcription.service`:
  - TranscriptionService

- `src_B.diarization.service`:
  - DiarizationService
  - DiarizationSegment
  - MAX_SPEAKERS

- `src_B.evaluation.evaluator`:
  - OllamaEvaluator

- `src_B.reports.generator`:
  - finalize_meeting_report

---

## Usage Example

```python
import uvicorn
from src_B.api.app import app

if __name__ == "__main__":
    uvicorn.run(
        "src_B.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
```

---

## API Testing

### Start Meeting
```bash
curl -X POST http://localhost:8000/api/start \
  -H "Content-Type: application/json" \
  -d '{"topic": "Team Standup", "expected_speakers": 3}'
```

### Get Status
```bash
curl http://localhost:8000/api/status
```

### Stop Meeting
```bash
curl -X POST http://localhost:8000/api/stop/{session_id}
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

---

## Notes

1. **Missing Method**: The original code calls `transcription_service.attach_diarization()` but this method was not found in the original file. The code includes conditional checks (`if hasattr(...)`) to handle this gracefully.

2. **Thread Safety**: All state modifications use proper locking mechanisms (`meeting_state_lock`, `meeting_history_lock`, etc.)

3. **Async/Sync Bridge**: Uses `loop.run_in_executor()` to call synchronous service methods from async routes

4. **Configuration**: Reads from environment variables:
   - `OLLAMA_GEMMA_MODEL` (default: "gemma3-270m-local-e3")
   - `OLLAMA_BASE_URL` (default: "http://localhost:11434")

---

## Validation

All Python files have been validated for syntax errors:
```bash
python3 -m py_compile src_B/api/*.py
# No errors found
```

**Total Lines of Code**: 904 lines across 4 files
