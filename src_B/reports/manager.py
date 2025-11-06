"""
Meeting report management and finalization.

This module coordinates the report generation process by combining
summary generation and PDF rendering into a complete workflow.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src_B.reports.constants import REPORTS_DIR
from src_B.reports.pdf_generator import render_pdf_report
from src_B.reports.summarizer import create_summary_with_gemma

logger = logging.getLogger(__name__)


def finalize_meeting_report(
    session_id: Optional[str],
    topic: str,
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Finalize and generate a complete meeting report.

    This function orchestrates the entire report generation process:
    1. Generates a meeting summary using the Gemma model
    2. Renders the summary as a PDF report
    3. Optionally notifies listeners via event callback

    Args:
        session_id: Unique identifier for the meeting session.
            If None, report generation is skipped.
        topic: The meeting topic/subject
        overall_stats: Overall meeting statistics dictionary containing:
            - avg_topic_relevance: Average topic relevance score
            - avg_novelty: Average novelty score
            - total_statements: Total number of statements
        speaker_stats: Per-speaker statistics dictionary with speaker IDs as keys
        history: List of meeting statement dictionaries
        event_callback: Optional callback function to notify about report status.
            Called with a dict containing:
            - type: "report_ready"
            - available: bool (True if successful)
            - path: str (report path if successful)
            - detail: str (error message if failed)

    Returns:
        Tuple of (report_path, summary_payload):
            - report_path: Path to generated PDF file, or None if failed
            - summary_payload: Generated summary dictionary, or None if failed

    Example:
        >>> def my_callback(event):
        ...     print(f"Report status: {event}")
        >>> path, summary = finalize_meeting_report(
        ...     "session123",
        ...     "Project Planning",
        ...     {"avg_topic_relevance": 8.5, "total_statements": 45},
        ...     {"Speaker1": {"avg_topic_relevance": 8.5}},
        ...     [{"timestamp": "10:30", "speaker_id": "Speaker1", ...}],
        ...     event_callback=my_callback
        ... )
    """
    if session_id is None:
        logger.warning("세션 ID 없이 보고서를 생성할 수 없습니다")
        return None, None

    # Generate summary using Gemma model
    summary = create_summary_with_gemma(topic, overall_stats, speaker_stats, history)

    # Attempt to render PDF report
    try:
        report_path = REPORTS_DIR / f"meeting_report_{session_id}.pdf"
        render_pdf_report(summary, overall_stats, speaker_stats, history, report_path)
        logger.info("PDF 보고서 생성 완료: %s", report_path)

        # Notify success via callback if provided
        if event_callback:
            event_callback(
                {
                    "type": "report_ready",
                    "available": True,
                    "path": str(report_path),
                }
            )

        return report_path, summary

    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF 보고서 생성 실패: %s", exc)

        # Notify failure via callback if provided
        if event_callback:
            event_callback(
                {
                    "type": "report_ready",
                    "available": False,
                    "detail": str(exc),
                }
            )

        return None, summary
