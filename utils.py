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
    """'Step 1: ...\\nStep 2: ...' 형식의 풀이를 step list로 분리."""
    lines = solution_text.strip().split("\n")
    steps: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^Step\s+\d+[:.]", line.strip()):
            if current:
                steps.append(" ".join(current).strip())
                current = []
        current.append(line.strip())
    if current:
        steps.append(" ".join(current).strip())
    return [s for s in steps if s]


def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K answer 필드에서 '#### 정답' 패턴 추출."""
    m = re.search(r"####\s*(.+?)$", answer_text.strip(), re.MULTILINE)
    return m.group(1).strip() if m else ""
