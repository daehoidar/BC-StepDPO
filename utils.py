"""utils.py — 데이터 파이프라인 공용 헬퍼.

- load_personas: enriched personas.json 로더
- parse_steps: 'Step 1: ... Step 2: ...' 형식 풀이를 step 리스트로 분해
- extract_gsm8k_answer: GSM8K answer 필드에서 #### 뒤 정답 추출
"""
from __future__ import annotations
import json
import re
from pathlib import Path


def load_personas(personas_path: str | Path) -> list[dict]:
    """personas.json -> 페르소나 리스트.

    enriched 상태(exemplar_standards + term_evidence 포함)를 기대한다.
    derive_persona_evidence.py를 먼저 한 번 실행해두어야 한다.
    """
    with open(personas_path, encoding="utf-8") as f:
        return json.load(f)["personas"]


def parse_steps(solution_text: str) -> list[str]:
    """'Step 1: ... Step 2: ...' 형식의 풀이를 step list로 분리.

    줄바꿈 유무와 무관하게 'Step N:'/'Step N.' 마커 기준으로 분할(inline 포함).
    (이전엔 줄바꿈으로만 잘라 inline 다단계 출력이 1 step으로 집계되던 버그)
    """
    text = solution_text.strip()
    if not text:
        return []
    # 각 'Step N:' 마커 앞에서 분할 → 마커로 시작하는 조각만 step으로 채택
    parts = re.split(r"(?=Step\s+\d+\s*[:.])", text)
    steps = [p.strip() for p in parts if re.match(r"Step\s+\d+\s*[:.]", p.strip())]
    return steps if steps else [text]  # 마커 없으면 단일 step


def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K answer 필드에서 '#### 정답' 패턴 추출."""
    m = re.search(r"####\s*(.+?)$", answer_text.strip(), re.MULTILINE)
    return m.group(1).strip() if m else ""
