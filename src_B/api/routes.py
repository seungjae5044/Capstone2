"""
FastAPI route handlers for meeting control and status endpoints.

This module provides REST API endpoints for:
- Health checks
- Meeting session control (start/stop)
- Speaker and timeline data
- Statistics and reports
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from src_B.api.state import (
    meeting_state,
    meeting_state_lock,
    meeting_stats,
    meeting_history,
    meeting_history_lock,
    transcription_service,
    diarization_service,
    ollama_evaluator,
    last_report_path,
    last_summary_payload,
    broadcast_session_status,
    broadcast_stats_messages,
    handle_transcription,
    ensure_diarization_service,
    handle_diarization_segment,
    update_active_speaker,
    enqueue_event,
)
from src_B.diarization.service import MAX_SPEAKERS
from src_B.reports.generator import finalize_meeting_report

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/api/speakers")
async def get_speakers() -> Dict[str, Any]:
    """Get speaker summaries and statistics."""
    return {
        "is_active": meeting_state.is_active,
        "speakers": transcription_service.speaker_summaries(),
        "stats": meeting_stats.speaker_dict(),
        "stats_overall": meeting_stats.overall_dict(),
    }


@router.get("/api/timeline")
async def get_timeline() -> Dict[str, Any]:
    """Get timeline of speaker segments."""
    return {
        "is_active": meeting_state.is_active,
        "segments": transcription_service.timeline_snapshot(),
    }


@router.post("/api/start")
async def start_meeting(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start a new meeting session.

    Args:
        payload: Dictionary containing:
            - topic (str): Meeting topic (required)
            - speaker_id (str): Initial speaker ID (default: "Speaker 1")
            - expected_speakers (int): Number of expected speakers (default: 2)

    Returns:
        Dictionary with success status, session_id, and topic

    Raises:
        HTTPException: If meeting is already active, topic is missing, or startup fails
    """
    if meeting_state.is_active:
        raise HTTPException(status_code=400, detail="Meeting is already active")

    # Parse and validate parameters
    topic = (payload.get("topic") or "").strip()
    speaker_id = (payload.get("speaker_id") or "Speaker 1").strip() or "Speaker 1"
    expected_speakers_raw = payload.get("expected_speakers")

    try:
        expected_speakers = int(expected_speakers_raw)
    except (TypeError, ValueError):
        expected_speakers = 2
    expected_speakers = max(1, min(expected_speakers, MAX_SPEAKERS))

    if not topic:
        raise HTTPException(status_code=400, detail="Meeting topic is required")

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]
    meeting_state.session_id = session_id
    meeting_state.last_session_id = session_id
    meeting_state.topic = topic
    meeting_state.speaker_id = speaker_id

    # Reset state
    meeting_stats.reset()
    ollama_evaluator.initialize(topic)

    global last_report_path, last_summary_payload
    with meeting_history_lock:
        meeting_history.clear()
    last_report_path = None
    last_summary_payload = None

    # Initialize services
    loop = asyncio.get_running_loop()
    diar_service: Optional[Any] = None

    try:
        await loop.run_in_executor(None, transcription_service.initialize)
        diar_service = await loop.run_in_executor(
            None,
            lambda: ensure_diarization_service(
                transcription_service.device,
                expected_speakers,
            ),
        )
        diar_service.set_segment_callback(handle_diarization_segment)
        diar_service.set_active_speaker_callback(update_active_speaker)
        diar_service.start()

        # Attach diarization to transcription service if method exists
        if hasattr(transcription_service, 'attach_diarization'):
            transcription_service.attach_diarization(diar_service)

        await loop.run_in_executor(None, lambda: transcription_service.start(handle_transcription))
    except Exception as exc:  # noqa: BLE001
        # Rollback state on error
        meeting_state.session_id = None
        meeting_state.last_session_id = None
        meeting_state.topic = ""
        meeting_state.speaker_id = "Speaker 1"
        logger.exception("Failed to start transcription: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start transcription. Check logs.")

    # Update active state and broadcast
    with meeting_state_lock:
        meeting_state.is_active = True

    broadcast_session_status()
    broadcast_stats_messages()
    logger.info("Meeting session started: %s (topic=%s)", session_id, topic)

    return {"success": True, "session_id": session_id, "topic": topic}


@router.post("/api/stop/{session_id}")
async def stop_meeting(session_id: str) -> Dict[str, Any]:
    """
    Stop the current meeting session.

    Args:
        session_id: Session ID to stop

    Returns:
        Dictionary with success status and message

    Raises:
        HTTPException: If session is not active or session ID doesn't match
    """
    if not meeting_state.is_active:
        if session_id == meeting_state.last_session_id:
            return {"success": True, "message": "Meeting already stopped"}
        raise HTTPException(status_code=400, detail="No active session")

    if meeting_state.session_id != session_id:
        if session_id == meeting_state.last_session_id:
            return {"success": True, "message": "Meeting already stopped"}
        raise HTTPException(status_code=400, detail="Session ID mismatch")

    # Capture state before stopping
    loop = asyncio.get_running_loop()
    session_id_current = meeting_state.session_id
    topic = meeting_state.topic
    overall_snapshot = meeting_stats.overall_dict()
    speaker_snapshot = meeting_stats.speaker_dict()

    with meeting_history_lock:
        history_snapshot = list(meeting_history)

    # Stop transcription service
    try:
        await loop.run_in_executor(None, transcription_service.stop)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to stop transcription: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to stop transcription")

    # Stop diarization service
    diar_service = diarization_service
    if diar_service is not None:
        await loop.run_in_executor(None, diar_service.stop)

    # Detach diarization from transcription service if method exists
    if hasattr(transcription_service, 'attach_diarization'):
        transcription_service.attach_diarization(None)

    # Update state
    with meeting_state_lock:
        meeting_state.is_active = False

    broadcast_session_status()

    # Clear visualization
    enqueue_event(
        {
            "type": "diarization",
            "speaker_id": "Standby",
            "is_pending": False,
            "similarity": 0.0,
            "offset": 0.0,
            "duration": 0.0,
        }
    )
    enqueue_event(
        {
            "type": "waveform",
            "points": [],
            "window_seconds": 0.0,
        }
    )

    # Generate report
    await loop.run_in_executor(
        None,
        lambda: finalize_meeting_report(
            session_id_current,
            topic,
            overall_snapshot,
            speaker_snapshot,
            history_snapshot,
        ),
    )

    broadcast_stats_messages()
    logger.info("Meeting session stopped: %s", session_id)

    # Reset state
    meeting_state.last_session_id = session_id_current
    meeting_state.session_id = None
    meeting_state.topic = ""
    meeting_state.speaker_id = "Speaker 1"

    return {"success": True, "message": "Meeting stopped"}


@router.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get current meeting status."""
    return {
        "is_active": meeting_state.is_active,
        "session_id": meeting_state.session_id,
        "topic": meeting_state.topic,
        "speaker_id": meeting_state.speaker_id,
    }


@router.get("/api/report")
async def get_report(format: str = "pdf") -> Any:  # noqa: ANN401
    """
    Get meeting report in PDF or JSON format.

    Args:
        format: Output format, either "pdf" or "json" (default: "pdf")

    Returns:
        FileResponse for PDF, JSONResponse for JSON

    Raises:
        HTTPException: If report is not available
    """
    if format == "json":
        if last_summary_payload:
            return JSONResponse(content=last_summary_payload)
        raise HTTPException(status_code=404, detail="Summary not yet generated")

    if last_report_path and last_report_path.exists():
        return FileResponse(
            path=last_report_path,
            media_type="application/pdf",
            filename=last_report_path.name,
        )

    if last_summary_payload:
        return JSONResponse(
            status_code=202,
            content={
                "detail": "PDF not ready yet. Request JSON summary with format=json.",
            },
        )

    raise HTTPException(status_code=404, detail="Report not yet generated")
