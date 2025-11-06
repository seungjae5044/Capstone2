"""
Meeting summary generation using Gemma model.

This module provides functionality to generate comprehensive meeting summaries
using the Ollama Gemma model, including speaker analysis and key insights.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from src_B.evaluation.evaluator import GEMMA_SYSTEM_PROMPT, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def _extract_json_object(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from response text.

    Finds the first '{' and last '}' in the response and attempts to
    parse the content between them as JSON.

    Args:
        response_text: Raw text response that may contain JSON

    Returns:
        Parsed JSON dictionary if successful, None otherwise

    Example:
        >>> _extract_json_object('Here is data: {"key": "value"} done')
        {"key": "value"}
    """
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = response_text[start : end + 1]
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def create_summary_with_gemma(
    topic: str,
    overall_stats: Dict[str, Any],
    speaker_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a comprehensive meeting summary using Gemma model.

    Analyzes meeting history and statistics to create a detailed report
    including overview, key points, speaker analysis, and recommendations.

    Args:
        topic: The meeting topic/subject
        overall_stats: Overall meeting statistics dictionary containing:
            - avg_topic_relevance: Average topic relevance score
            - avg_novelty: Average novelty score
            - total_statements: Total number of statements
        speaker_stats: Per-speaker statistics dictionary with speaker IDs as keys
            and their individual stats as values
        history: List of meeting statement dictionaries, each containing:
            - timestamp: When the statement was made
            - speaker_id: ID of the speaker
            - text: Statement text
            - topic_relevance: Topic relevance score
            - novelty: Novelty score

    Returns:
        Dictionary containing the meeting summary with keys:
            - title: Meeting title
            - overview: Overall meeting summary
            - key_points: List of key discussion points
            - insights: List of key insights
            - speaker_analysis: List of per-speaker analysis dictionaries
            - metrics: Meeting metrics dictionary
            - recommendations: List of improvement recommendations
            - conclusion: Final conclusion

    Example:
        >>> summary = create_summary_with_gemma(
        ...     "Project Planning",
        ...     {"avg_topic_relevance": 8.5, "avg_novelty": 7.0, "total_statements": 45},
        ...     {"Speaker1": {"avg_topic_relevance": 8.5, "total_statements": 20}},
        ...     [{"timestamp": "10:30", "speaker_id": "Speaker1", "text": "...", ...}]
        ... )
    """
    if not history:
        return {
            "title": f"{topic} 회의 요약",
            "overview": "회의 발언 데이터가 없어 요약을 생성하지 못했습니다.",
            "key_points": [],
            "insights": [],
            "speaker_analysis": [],
            "metrics": overall_stats,
            "recommendations": [],
            "conclusion": "",
        }

    recent_history = history[-40:]
    history_lines = [
        (
            f"- {item['timestamp']} | {item['speaker_id']}: {item['text']}"
            f" (주제 일치 {item['topic_relevance']:.1f}, 신규성 {item['novelty']:.1f})"
        )
        for item in recent_history
    ]

    prompt = (
        f"{GEMMA_SYSTEM_PROMPT}\n\n"
        "다음 회의 기록을 바탕으로 전문적인 분석 보고서를 작성하세요."
        "보고서는 JSON 형식으로 응답하며, 키는 다음과 같습니다:\n"
        "{\n"
        "  \"title\": 회의 제목 문자열,\n"
        "  \"overview\": 회의 전반 요약 (문단),\n"
        "  \"key_points\": 주요 논의 사항 목록,\n"
        "  \"insights\": 핵심 인사이트 목록,\n"
        "  \"speaker_analysis\": [ {\"speaker\", \"summary\", \"topic_score\", \"novelty_score\", \"contribution\"} ],\n"
        "  \"metrics\": {\"avg_topic\", \"avg_novelty\", \"total_statements\", \"participation_level\"},\n"
        "  \"recommendations\": 개선 제안 목록,\n"
        "  \"conclusion\": 결론 요약\n"
        "}\n"
        "JSON 이외의 텍스트는 포함하지 마세요.\n\n"
        f"회의 주제: {topic}\n"
        f"전체 통계: {json.dumps(overall_stats, ensure_ascii=False)}\n"
        f"화자별 통계: {json.dumps(speaker_stats, ensure_ascii=False)}\n"
        "최근 발언 로그:\n"
        f"{os.linesep.join(history_lines)}\n\n"
        "보고서를 생성하세요."
    )

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
        parsed = _extract_json_object(raw)
        if parsed:
            return parsed
        logger.warning("Gemma 응답에서 JSON을 추출하지 못했습니다. 원문: %s", raw[:200])
    except requests.exceptions.RequestException as exc:
        logger.error("Gemma 요약 생성 실패: %s", exc)

    return {
        "title": f"{topic} 회의 요약",
        "overview": "Gemma 요약 생성에 실패하여 기본 통계만 제공합니다.",
        "key_points": [],
        "insights": [],
        "speaker_analysis": [
            {
                "speaker": speaker,
                "summary": "요약 생성 실패",
                "topic_score": stats.get("avg_topic_relevance", 0.0),
                "novelty_score": stats.get("avg_novelty", 0.0),
                "contribution": stats.get("total_statements", 0),
            }
            for speaker, stats in speaker_stats.items()
        ],
        "metrics": {
            "avg_topic": overall_stats.get("avg_topic_relevance", 0.0),
            "avg_novelty": overall_stats.get("avg_novelty", 0.0),
            "total_statements": overall_stats.get("total_statements", 0),
            "participation_level": "",
        },
        "recommendations": [],
        "conclusion": "",
    }
