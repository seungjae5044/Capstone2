"""
Report generation module for meeting summaries.

This package provides comprehensive report generation capabilities including:
- Summary generation using Gemma AI model
- Professional PDF report rendering
- Complete report workflow management

Main Components:
    - constants: Report-related configuration constants
    - summarizer: AI-powered meeting summary generation
    - pdf_generator: PDF report rendering
    - manager: Complete report workflow orchestration

Example:
    >>> from src_B.reports import finalize_meeting_report
    >>> path, summary = finalize_meeting_report(
    ...     session_id="meeting123",
    ...     topic="Project Planning",
    ...     overall_stats={...},
    ...     speaker_stats={...},
    ...     history=[...]
    ... )
"""
from src_B.reports.constants import REPORTS_DIR
from src_B.reports.manager import finalize_meeting_report
from src_B.reports.pdf_generator import REPORTLAB_AVAILABLE, render_pdf_report
from src_B.reports.summarizer import create_summary_with_gemma

__all__ = [
    "REPORTS_DIR",
    "REPORTLAB_AVAILABLE",
    "create_summary_with_gemma",
    "render_pdf_report",
    "finalize_meeting_report",
]
