"""
PDF report generation for meeting summaries.

This module provides functionality to generate professionally formatted
PDF reports from meeting summaries using the ReportLab library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Optional reportlab dependency
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False


def render_pdf_report(
    summary: Dict[str, Any],
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a professionally formatted PDF report from meeting summary.

    Creates a multi-section PDF report with styled tables and paragraphs
    including meeting overview, speaker analysis, statistics, and recommendations.

    Args:
        summary: Meeting summary dictionary containing:
            - title: Report title
            - overview: Meeting overview text
            - key_points: List of key discussion points
            - insights: List of key insights
            - speaker_analysis: List of speaker analysis dictionaries
            - metrics: Meeting metrics dictionary
            - recommendations: List of recommendations
            - conclusion: Final conclusion text
        overall_stats: Overall meeting statistics (currently unused in PDF)
        speaker_stats: Per-speaker statistics (currently unused in PDF)
        history: List of meeting statements for appendix (last 20 shown)
        output_path: Path where the PDF file should be saved

    Raises:
        RuntimeError: If reportlab library is not installed

    Note:
        Requires reportlab library. Install with: pip install reportlab

    Example:
        >>> from pathlib import Path
        >>> summary = {
        ...     "title": "Meeting Report",
        ...     "overview": "Summary of discussion...",
        ...     "key_points": ["Point 1", "Point 2"],
        ...     ...
        ... }
        >>> render_pdf_report(summary, {}, {}, [], Path("report.pdf"))
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab 라이브러리가 설치되어 있지 않습니다. 'pip install reportlab' 후 다시 시도하세요.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=1,
        fontSize=18,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        textColor="white",
        backColor="#1F4E79",
        alignment=0,
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        leftIndent=4,
        rightIndent=4,
        leading=14,
    )
    text_style = styles["BodyText"]
    text_style.leading = 14

    story: List[Any] = []
    story.append(Paragraph(summary.get("title", "회의 보고서"), title_style))
    story.append(Paragraph("회의 개요", section_style))
    story.append(Paragraph(summary.get("overview", ""), text_style))

    if summary.get("key_points"):
        story.append(Paragraph("주요 논의 사항", section_style))
        for idx, point in enumerate(summary["key_points"], start=1):
            story.append(Paragraph(f"{idx}. {point}", text_style))
            story.append(Spacer(1, 4))

    if summary.get("insights"):
        story.append(Paragraph("핵심 인사이트", section_style))
        for idx, insight in enumerate(summary["insights"], start=1):
            story.append(Paragraph(f"{idx}. {insight}", text_style))
            story.append(Spacer(1, 4))

    speaker_data = summary.get("speaker_analysis") or []
    if speaker_data:
        story.append(Paragraph("발언자별 평가", section_style))
        table_data = [["화자", "요약", "주제 점수", "신규성", "기여"]]
        for item in speaker_data:
            table_data.append(
                [
                    item.get("speaker", "-"),
                    item.get("summary", ""),
                    str(item.get("topic_score", "")),
                    str(item.get("novelty_score", "")),
                    str(item.get("contribution", "")),
                ]
            )

        table = Table(table_data, colWidths=[25 * mm, 65 * mm, 20 * mm, 20 * mm, 20 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E1F2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.gray),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.gray),
                ]
            )
        )
        story.append(table)

    metrics = summary.get("metrics", {})
    if metrics:
        story.append(Paragraph("회의 통계", section_style))
        metrics_table = Table(
            [
                ["평균 주제일치성", str(metrics.get("avg_topic", "")), "평균 신규성", str(metrics.get("avg_novelty", ""))],
                ["총 발언 수", str(metrics.get("total_statements", "")), "참여도", str(metrics.get("participation_level", ""))],
            ],
            colWidths=[35 * mm, 35 * mm, 35 * mm, 35 * mm],
        )
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E9EDF5")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F4E79")),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.gray),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.gray),
                ]
            )
        )
        story.append(metrics_table)

    if summary.get("recommendations"):
        story.append(Paragraph("결론 및 제안", section_style))
        for idx, rec in enumerate(summary["recommendations"], start=1):
            story.append(Paragraph(f"{idx}. {rec}", text_style))
            story.append(Spacer(1, 4))

    if summary.get("conclusion"):
        story.append(Paragraph("최종 결론", section_style))
        story.append(Paragraph(summary["conclusion"], text_style))

    if history:
        story.append(Paragraph("부록 - 발언 로그", section_style))
        for item in history[-20:]:
            story.append(
                Paragraph(
                    f"{item['timestamp']} | {item['speaker_id']} : {item['text']} (주제 {item['topic_relevance']:.1f}, 신규성 {item['novelty']:.1f})",
                    text_style,
                )
            )

    doc.build(story)
