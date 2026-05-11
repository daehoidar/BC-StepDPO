"""부트스트랩 합성 결과에서 페르소나별 베스트 풀이를 골라 fewshot.json으로 저장한다.

워크플로:
1. 1_synthesize_sft.py를 --limit 300 등으로 소규모 실행 -> bootstrap.jsonl
2. 이 스크립트를 실행하면 페르소나별 후보를 보여주고 인덱스 입력을 받는다
3. 선택 결과가 curriculum/fewshot.json에 저장됨
4. 다음 번 1_synthesize_sft.py 실행 시 personas.py가 자동으로 로드하여 적용

사용 예:
    python data_pipeline/1b_pick_fewshot.py \
        --input data_pipeline/output/bootstrap.jsonl

옵션:
    --auto-first  인덱스 0번을 자동 선택 (검토 없이 빠르게 진행)
    --output      저장 경로 (기본: curriculum/fewshot.json)
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personas import PERSONA_GRID  # noqa: E402


def load_bootstrap(path: Path) -> dict:
    """jsonl을 페르소나별로 그룹핑."""
    by_persona = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "gpt4o_solution" not in r:
                continue  # 실패한 행 스킵
            by_persona[r["persona"]].append(r)
    return by_persona


def print_candidates(pid: str, items: list[dict]) -> None:
    print()
    print("=" * 72)
    print(f"  PERSONA: {pid}   (후보 {len(items)}개)")
    print("=" * 72)
    for i, it in enumerate(items):
        print()
        print(f"--- [{i}]  problem_id={it['problem_id']}  difficulty={it.get('difficulty','?')}")
        print(f"Q: {it['question']}")
        print()
        print(f"A:\n{it['gpt4o_solution']}")
        print("-" * 72)


def ask_index(pid: str, n: int, auto_first: bool) -> int:
    if auto_first:
        print(f"[auto] {pid}: 인덱스 0 자동 선택")
        return 0
    while True:
        s = input(f"\n[{pid}] 선택할 인덱스 (0~{n-1}, 'skip' 입력 시 건너뜀): ").strip()
        if s.lower() == "skip":
            return -1
        try:
            i = int(s)
            if 0 <= i < n:
                return i
        except ValueError:
            pass
        print(f"  유효하지 않은 입력. 0~{n-1} 또는 'skip' 입력.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="부트스트랩 jsonl 경로")
    ap.add_argument("--output", default=str(REPO_ROOT / "curriculum" / "fewshot.json"))
    ap.add_argument("--auto-first", action="store_true",
                    help="인덱스 0번을 모든 페르소나에 자동 적용 (테스트용)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[error] not found: {in_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_persona = load_bootstrap(in_path)
    if not by_persona:
        sys.exit("[error] 부트스트랩 결과에서 유효한 행을 찾지 못함.")

    # 기존 fewshot.json이 있으면 로드해서 부분 갱신 가능
    picks: dict = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            picks = json.load(f)
        print(f"[load] 기존 fewshot.json 발견. 기존 항목 위에 덮어씁니다 ({len(picks)}개).")

    # PERSONA_GRID 순서대로 진행
    for age, level in PERSONA_GRID:
        pid = f"{age}-{level}"
        items = by_persona.get(pid, [])
        if not items:
            print(f"\n[warn] {pid}: 후보 없음. 스킵.")
            continue
        print_candidates(pid, items)
        idx = ask_index(pid, len(items), args.auto_first)
        if idx < 0:
            print(f"  -> {pid} 스킵")
            continue
        chosen = items[idx]
        picks[pid] = {
            "problem_id": chosen["problem_id"],
            "question": chosen["question"],
            "solution": chosen["gpt4o_solution"],
        }
        print(f"  -> {pid}: [{idx}] {chosen['problem_id']} 저장")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(picks, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 페르소나 {len(picks)}개 저장 -> {out_path}")


if __name__ == "__main__":
    main()
