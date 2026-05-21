"""MetaMathQA-40K에서 문제를 샘플링하여 페르소나 6종 공통 풀로 배정한다.

설계:
- MetaMathQA-40K는 GSM8K + MATH 두 원본을 4가지 방식으로 augmentation:
  AnsAug (답안 변형, 같은 query / 다른 response),
  Rephrased (문장 변형, query·response 모두 다름),
  FOBAR (forward-to-backward, query 구조 변환),
  SV (self-verification 형식 변환).
- 본 파이프라인은 **GSM8K 계열만** 사용: type 컬럼이 'GSM_'로 시작하는 행만 채택.
- AnsAug는 *같은 query 위에 다른 response*를 붙인 형태라, 학습 데이터 중복을
  유발한다. **query 기준 dedupe로 unique한 문제만** 채택.
  (Rephrased는 query가 달라 다른 행으로 살아남음.)
- 난이도 버킷팅은 response 내 <<...>> 마커 개수 + question 길이로 추정.
  GSM8K augmentation은 보통 GSM8K 형식의 <<...>> 마커를 보존.
- 같은 문제를 6 페르소나 모두에 복제 배정 (belief_pair 단계의 cross-persona
  비교 자연 발생을 위함).

출력: data_pipeline/output/seed_problems.jsonl
한 행 = (문제, 페르소나) 한 쌍. 총 행 수 = N × 6.

사용 예:
    python data_pipeline/0_seed_problems.py --n-problems 1500 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personas import PERSONA_GRID  # noqa: E402

# 공통 풀에 포함할 버킷 (hard는 제외해 페르소나 6종 모두가 풀 만한 수준 확보)
COMMON_BUCKETS = ["easy", "medium"]

OPS_RE = re.compile(r"<<[^>]+>>")
# MetaMathQA response 끝에 "The answer is: X" 또는 "The answer is X" 형태로 정답 표기
ANS_RE = re.compile(r"The answer is:?\s*([^\.\n]+?)\.?\s*$", re.MULTILINE)


def difficulty_bucket(query: str, response: str) -> str:
    """response의 <<...>> 마커 개수 + query 길이로 난이도 추정.

    GSM8K augmentation은 보통 원본의 <<X+Y=Z>> 형식을 보존한다. Rephrased·FOBAR·SV
    변형도 추론 과정에서 마커를 그대로 두는 경우가 많다. 마커가 없으면 query 길이만
    참조하고 보수적으로 medium 이상으로 분류.
    """
    n_ops = len(OPS_RE.findall(response))
    q_words = len(query.split())
    if n_ops > 0:
        if n_ops <= 2 and q_words <= 30:
            return "easy"
        if n_ops <= 4 and q_words <= 60:
            return "medium"
        return "hard"
    # 마커가 없으면 query 길이만으로 보수적 분류
    if q_words <= 30:
        return "medium"
    return "hard"


def extract_gt_answer(response: str) -> str:
    """'The answer is: X' 또는 'The answer is X' 패턴에서 X 추출.

    response 끝에 위 패턴이 없으면 마지막 줄에서 숫자/표현 휴리스틱 추출.
    """
    m = ANS_RE.search(response)
    if m:
        return m.group(1).strip().rstrip(".")
    # fallback: 마지막 줄
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def load_metamath():
    from datasets import load_dataset
    return load_dataset("meta-math/MetaMathQA-40K", split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=1500,
                    help="공통 풀에서 뽑을 문제 개수 (각 문제는 6 페르소나 모두에 배정)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(REPO_ROOT / "data_pipeline" / "output" / "seed_problems.jsonl"))
    ap.add_argument("--include-math", action="store_true",
                    help="GSM_* 외에 MATH_*도 포함 (기본은 GSM8K 계열만)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[load] MetaMathQA-40K")
    ds = load_metamath()
    print(f"[load] {len(ds)} rows (raw)")

    # 소스 데이터셋 필터 (기본: GSM 계열만)
    prefix_allow = ("GSM_",) if not args.include_math else ("GSM_", "MATH_")
    filtered = [r for r in ds if r.get("type", "").startswith(prefix_allow)]
    by_type: dict[str, int] = {}
    for r in filtered:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"[filter] allowed={prefix_allow} -> {len(filtered)} rows")
    for t, n in sorted(by_type.items()):
        print(f"  - {t}: {n}")

    # query 기준 dedupe (AnsAug 같은 답안 변형 중복 제거)
    seen_first: dict[str, dict] = {}
    for r in filtered:
        q = r["query"]
        if q not in seen_first:
            seen_first[q] = r
    unique_rows = list(seen_first.values())
    print(f"[dedupe] unique query: {len(unique_rows)} "
          f"(removed {len(filtered) - len(unique_rows)} dup queries)")

    # 버킷 분류
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, it in enumerate(unique_rows):
        b = difficulty_bucket(it["query"], it.get("response", ""))
        buckets[b].append({
            "problem_id": f"metamath_{idx}",
            "question": it["query"],
            "gt_answer_raw": it["response"],
            "gt_answer": extract_gt_answer(it["response"]),
            "augmentation_type": it.get("type", ""),
            "difficulty": b,
            "n_ops": len(OPS_RE.findall(it.get("response", ""))),
        })
    for b, lst in buckets.items():
        print(f"[bucket] {b}: {len(lst)}")

    # 공통 풀 구성
    common_pool = []
    for b in COMMON_BUCKETS:
        common_pool.extend(buckets[b])
    print(f"[common pool] {len(common_pool)} (buckets={COMMON_BUCKETS})")

    random.shuffle(common_pool)
    picked = common_pool[: args.n_problems]
    if len(picked) < args.n_problems:
        print(f"[warn] 요청 {args.n_problems}개 대비 풀 크기 {len(picked)}개. 가능한 만큼만 사용.")
    print(f"[pick] {len(picked)}개 문제 선정")

    # 6 페르소나 복제 배정
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in picked:
            for age, level in PERSONA_GRID:
                pid = f"{age}-{level}"
                row = {
                    "problem_id": item["problem_id"],
                    "persona": pid,
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "gt_answer_raw": item["gt_answer_raw"],
                    "difficulty": item["difficulty"],
                    "augmentation_type": item["augmentation_type"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_total += 1
    print(f"\n[done] 문제 {len(picked)} × 페르소나 {len(PERSONA_GRID)} = {n_total}행")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
