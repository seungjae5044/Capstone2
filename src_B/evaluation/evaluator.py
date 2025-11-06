"""
Ollama-based evaluation module for meeting transcriptions.

This module provides the OllamaEvaluator class which uses Ollama's Gemma model
to evaluate speaker statements based on topic relevance and novelty.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma3-270m-local-e3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# System prompt for Gemma model evaluation
GEMMA_SYSTEM_PROMPT = (
    "당신은 전문적인 회의 평가 AI입니다.\n"
    "다음 기준으로 발언을 평가하세요:\n"
    "1. 주제일치성 (0-10): 회의 주제와의 관련성\n"
    "2. 신규성 (0-10): 새로운 정보나 관점 제공 정도\n"
    "3. 기여도: 회의 진행에 대한 기여 정도\n\n"
    "평가는 객관적이고 건설적이어야 하며, JSON 형식으로 응답하세요."
)


class OllamaEvaluator:
    """Ollama 서버의 Gemma 모델을 활용해 발언을 평가한다.

    변경된 모델 I/O에 맞춰 단일 프롬프트로 평가하며,
    화자별 누적 컨텍스트(speaker_context)를 유지한다.

    Attributes:
        model: Ollama 모델 이름
        base_url: Ollama 서버 기본 URL
        meeting_topic: 현재 회의 주제
        lock: 스레드 안전을 위한 락
        max_ctx_chars: 컨텍스트 최대 길이 (문자)

    Example:
        >>> evaluator = OllamaEvaluator()
        >>> evaluator.initialize("프로젝트 진행 상황 회의")
        >>> result = evaluator.evaluate("새로운 기능을 추가했습니다.", "speaker_1")
        >>> print(result)
        {'topic_relevance': 8.5, 'novelty': 7.0, 'comment': '프로젝트와 관련된 발언'}
    """

    def __init__(self, model: str = "gemma3-270m-local-e3", base_url: str = "http://localhost:11434") -> None:
        """OllamaEvaluator 초기화.

        Args:
            model: Ollama 모델 이름 (기본값: "gemma3-270m-local-e3")
            base_url: Ollama 서버 URL (기본값: "http://localhost:11434")
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.meeting_topic = ""
        self.lock = threading.Lock()
        # 화자별 이전 발언 누적 (현재 세션 한정)
        self._speaker_history: Dict[str, List[str]] = {}
        # 컨텍스트 최대 길이 (문자)
        self.max_ctx_chars: int = int(os.environ.get("SPEAKER_CONTEXT_CHARS", 400))

    def initialize(self, topic: str) -> None:
        """평가기를 새 회의 주제로 초기화하고 화자 히스토리를 클리어한다.

        Args:
            topic: 회의 주제
        """
        with self.lock:
            self.meeting_topic = topic
            self._speaker_history.clear()

    def _clamp_tail(self, text: str, limit: int) -> str:
        """텍스트를 지정된 길이로 제한하고 초과 시 끝부분만 유지한다.

        Args:
            text: 원본 텍스트
            limit: 최대 문자 수

        Returns:
            제한된 텍스트 (초과 시 앞에 '…' 추가)
        """
        if not text or limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return "…" + text[-(limit - 1) :]

    def _get_speaker_context_text(self, speaker_id: str) -> str:
        """특정 화자의 이전 발언 컨텍스트를 가져온다.

        Args:
            speaker_id: 화자 ID

        Returns:
            화자의 이전 발언을 결합한 컨텍스트 텍스트
        """
        history = self._speaker_history.get(speaker_id) or []
        context_text = " ".join(history).strip()
        return self._clamp_tail(context_text, self.max_ctx_chars)

    def _append_speaker_sentence(self, speaker_id: str, sentence: str) -> None:
        """화자의 발언을 히스토리에 추가한다.

        Args:
            speaker_id: 화자 ID
            sentence: 추가할 발언
        """
        if not sentence:
            return
        history = self._speaker_history.setdefault(speaker_id, [])
        history.append(sentence)
        # 간단한 메모리 제어: 너무 많은 항목이면 앞부분 제거
        if len(history) > 200:
            del history[: len(history) - 200]

    def _build_prompt(self, topic: str, speaker_id: str, speaker_context: str, sentence: str) -> str:
        """평가를 위한 프롬프트를 구성한다.

        Args:
            topic: 회의 주제
            speaker_id: 화자 ID
            speaker_context: 화자의 이전 발언 컨텍스트
            sentence: 평가할 현재 발언

        Returns:
            구성된 프롬프트 문자열
        """
        # 파인튜닝 데이터와 동일한 형식
        return (
            f"topic: {topic}\n"
            f"speaker_id: {speaker_id}\n"
            f"speaker_context: {speaker_context or '없음'}\n"
            f"sentence: {sentence}\n"
        )

    def _call_model(self, prompt: str, timeout: float = 60.0) -> str:
        """Ollama 모델 API를 호출한다.

        Args:
            prompt: 모델에 전달할 프롬프트
            timeout: 요청 타임아웃 (초)

        Returns:
            모델의 응답 텍스트 (실패 시 빈 문자열)
        """
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            logger.error("Ollama 요청 실패: %s", exc)
            return ""
        except ValueError as exc:
            logger.error("Ollama 응답 파싱 실패: %s", exc)
            return ""

    def _parse_reason_and_scores(self, response_text: str) -> Tuple[str, float, float]:
        """모델 응답에서 평가 이유와 점수를 파싱한다.

        응답 형식: "이유 텍스트 (topic_score, novelty_score)"
        예: "좋은 발언입니다 (8.5, 7.0)"

        Args:
            response_text: 모델의 응답 텍스트

        Returns:
            (이유, 주제일치성 점수, 신규성 점수) 튜플
        """
        text = (response_text or "").strip()
        if not text:
            return "", 5.0, 5.0

        # 괄호로 끝나는 "… (x, y)" 패턴 우선 추출
        m = re.search(r"\(([^)]*)\)\s*$", text)
        topic_score = 5.0
        novelty_score = 5.0
        reason = text
        if m:
            inside = m.group(1)
            # 쉼표 또는 공백 구분 허용
            parts = [p.strip() for p in re.split(r"[,\s]+", inside) if p.strip()]
            if len(parts) >= 2:
                try:
                    topic_score = float(parts[0])
                    novelty_score = float(parts[1])
                except ValueError:
                    pass
            # 괄호 제거한 나머지를 이유로 사용
            reason = text[: m.start()].rstrip()

        # 0~10 범위로 클램프
        topic_score = max(0.0, min(10.0, topic_score))
        novelty_score = max(0.0, min(10.0, novelty_score))
        return reason, topic_score, novelty_score

    def evaluate(self, sentence: str, speaker_id: str) -> Dict[str, Any]:
        """단일 프롬프트(주제/화자/컨텍스트/문장)로 평가 후 점수/코멘트 반환.

        Args:
            sentence: 평가할 발언 문장
            speaker_id: 발언자 ID

        Returns:
            평가 결과 딕셔너리:
            {
                "topic_relevance": float,  # 주제일치성 점수 (0-10)
                "novelty": float,          # 신규성 점수 (0-10)
                "comment": str             # 평가 코멘트
            }
        """
        # 현재 화자의 이전 발언으로 컨텍스트 구성
        with self.lock:
            topic = self.meeting_topic
            speaker_context = self._get_speaker_context_text(speaker_id)

        prompt = self._build_prompt(topic, speaker_id, speaker_context, sentence)
        raw = self._call_model(prompt)
        reason, topic_score, novelty_score = self._parse_reason_and_scores(raw)

        # 평가 이후 현재 문장을 화자 컨텍스트에 추가
        with self.lock:
            self._append_speaker_sentence(speaker_id, sentence)

        return {
            "topic_relevance": topic_score,
            "novelty": novelty_score,
            "comment": reason,
        }
