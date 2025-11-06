"""
Helper functions for evaluation processing.

This module provides utility functions for parsing and extracting
evaluation-related data from various sources.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _extract_json_object(response_text: str) -> Optional[Dict[str, Any]]:
    """응답 텍스트에서 JSON 객체를 추출한다.

    텍스트 내에서 첫 번째 '{' 와 마지막 '}' 사이의 문자열을
    JSON 객체로 파싱을 시도한다.

    Args:
        response_text: JSON 객체를 포함할 수 있는 응답 텍스트

    Returns:
        파싱된 JSON 객체 딕셔너리, 실패 시 None

    Example:
        >>> text = 'Some text {"score": 8.5, "comment": "Good"} more text'
        >>> result = _extract_json_object(text)
        >>> print(result)
        {'score': 8.5, 'comment': 'Good'}

        >>> text = 'No JSON here'
        >>> result = _extract_json_object(text)
        >>> print(result)
        None
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
