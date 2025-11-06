"""
FastAPI application setup for the meeting evaluation system.

This module:
- Creates and configures the FastAPI application
- Registers routes and WebSocket endpoints
- Manages startup and shutdown lifecycle events
- Initializes background workers for broadcasting and evaluation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI

from src_B.api.routes import router
from src_B.api.websocket import websocket_endpoint, broadcast_worker
from src_B.api import state
from src_B.evaluation.evaluator import OllamaEvaluator

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Whisper Meeting API",
    description="Real-time meeting transcription and evaluation API (No Web UI)",
    version="1.0.0",
)

# Include routes
app.include_router(router)


# Register WebSocket endpoint
@app.websocket("/ws")
async def ws_endpoint(websocket) -> None:  # noqa: ANN001
    """WebSocket endpoint for real-time updates."""
    await websocket_endpoint(websocket)


async def evaluation_worker() -> None:
    """
    Background worker for processing evaluation tasks.

    Continuously pulls tasks from the evaluation queue and processes them.
    """
    queue_: asyncio.Queue = app.state.evaluation_queue
    loop = asyncio.get_running_loop()

    while True:
        task = await queue_.get()
        try:
            await loop.run_in_executor(
                None,
                lambda t=task: _evaluate_and_broadcast(t),
            )
        finally:
            queue_.task_done()


def _evaluate_and_broadcast(task: Dict[str, Any]) -> None:
    """
    Evaluate a transcription and broadcast the results.

    Args:
        task: Evaluation task containing text, speaker_id, session_id, etc.
    """
    text = task.get("text", "")
    speaker_id = task.get("speaker_id", "Speaker 1")
    session_id = task.get("session_id")
    speaker_name = task.get("speaker_name", speaker_id)
    duration = float(task.get("duration", 0.0) or 0.0)
    timestamp_str = task.get("timestamp")

    if not text or session_id is None:
        return

    if not state.meeting_state.is_active or state.meeting_state.session_id != session_id:
        return

    # Evaluate the text
    try:
        result = state.ollama_evaluator.evaluate(text, speaker_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed: %s", exc)
        result = {"topic_relevance": 5.0, "novelty": 5.0, "comment": ""}

    # Update statistics
    state.meeting_stats.add_statement(speaker_id, result["topic_relevance"], result["novelty"])

    # Add to meeting history
    with state.meeting_history_lock:
        state.meeting_history.append(
            {
                "text": text,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "duration": duration,
                "topic_relevance": result["topic_relevance"],
                "novelty": result["novelty"],
                "timestamp": timestamp_str or "",
                "comment": result.get("comment", ""),
            }
        )

    # Broadcast evaluation result
    state.enqueue_event(
        {
            "type": "evaluation",
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "text": text,
            "topic_relevance": result["topic_relevance"],
            "novelty": result["novelty"],
            "timestamp": timestamp_str or "",
            "comment": result.get("comment", ""),
        }
    )

    # Broadcast updated statistics
    state.broadcast_stats_messages()


@app.on_event("startup")
async def on_startup() -> None:
    """
    Application startup handler.

    Initializes:
    - Event loop reference
    - Event and evaluation queues
    - Background workers for broadcasting and evaluation
    """
    app.state.event_loop = asyncio.get_running_loop()
    app.state.event_queue = asyncio.Queue()
    app.state.broadcast_task = asyncio.create_task(broadcast_worker(app.state.event_queue))
    app.state.evaluation_queue = asyncio.Queue()
    app.state.evaluation_task = asyncio.create_task(evaluation_worker())

    # Set app state reference in state module
    state.set_app_state(app.state)

    logger.info("Background workers started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """
    Application shutdown handler.

    Cancels and awaits background worker tasks.
    """
    for attr in ("broadcast_task", "evaluation_task"):
        task: Optional[asyncio.Task] = getattr(app.state, attr, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    logger.info("Background workers stopped")
