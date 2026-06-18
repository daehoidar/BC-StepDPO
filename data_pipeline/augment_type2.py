"""augment_type2.py — Type-2(belief-flip) 페어를 제약 완화로 증량 (B1).

기존 build_type2_pairs는 candidates[:2] + max_per_problem=3 으로 제한적.
이 스크립트는 Type-1 persona_first_error 페어에 대해 **모든 후보 페르소나**를
대상으로 cross-belief 체크를 **병렬** 수행해 추가 Type-2를 만든다.

입력: preference_pairs(또는 _aug).jsonl  (Type-1 step_pair + 기존 Type-2 포함)
출력: 원본 전체 + 신규 Type-2 (중복 제거) 합본.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils import load_personas          # noqa: E402
from openai_client import make_openai_client  # noqa: E402

# 3_build_pairs (숫자 시작 모듈명) → importlib
_spec = importlib.util.spec_from_file_location(
    "bp", str(REPO_ROOT / "data_pipeline" / "3_build_pairs.py"))
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)
call_cross_belief_check = bp.call_cross_belief_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--max-candidates", type=int, default=5,
                    help="cross-belief 비교할 다른 페르소나 수(기존 2 → 완화)")
    ap.add_argument("--max-per-problem", type=int, default=20,
                    help="문제당 Type-2 상한(기존 3 → 완화)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    personas = load_personas(args.personas_path)
    persona_by_id = {p["id"]: p for p in personas}
    client = make_openai_client()

    all_pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    type1 = [p for p in all_pairs if p.get("pair_type") == "step_pair"
             and p.get("reject_type") == "reject_persona"]
    existing_t2 = sum(1 for p in all_pairs if p.get("pair_type") == "belief_flip_pair")
    print(f"[load] {len(all_pairs)} pairs (Type-1 persona {len(type1)}, 기존 Type-2 {existing_t2})")

    # 작업 수집: (pair, candidate_persona)
    tasks = []
    for p in type1:
        cur = persona_by_id.get(p["persona_id"])
        if cur is None:
            continue
        cands = [o for o in personas if o["id"] != cur["id"]
                 and (o["grade_band"] != cur["grade_band"] or o["level"] != cur["level"])]
        for other in cands[: args.max_candidates]:
            tasks.append((p, cur, other))
    print(f"[tasks] {len(tasks)} cross-belief 체크 (workers={args.workers})")

    def _check(t):
        p, cur, other = t
        chk = call_cross_belief_check(
            client=client, model=args.gpt_model,
            step_text=p["step_lose"], prefix_text="\n".join(p.get("prefix_steps", [])),
            problem={"problem": p["problem"]}, persona_a=cur, persona_b=other)
        ok = (chk.get("flip") and not chk.get("persona_a_acceptable")
              and chk.get("persona_b_acceptable"))
        return (p, other, chk) if ok else None

    new_t2 = []
    per_problem = defaultdict(int)
    seen = set()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_check, tasks):
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(tasks)}  new Type-2={len(new_t2)}")
            if res is None:
                continue
            p, other, chk = res
            if per_problem[p["problem_id"]] >= args.max_per_problem:
                continue
            key = (p["problem_id"], p.get("step_lose", "")[:60], other["id"])
            if key in seen:
                continue
            seen.add(key); per_problem[p["problem_id"]] += 1
            new_t2.append({
                "problem_id": p["problem_id"], "problem": p["problem"],
                "persona_id": p["persona_id"], "persona_tag": p.get("persona_tag"),
                "prefix_steps": p.get("prefix_steps", []),
                "step_win": p["step_win"], "step_lose": p["step_lose"],
                "pair_type": "belief_flip_pair", "pair_subtype": "persona_first_error",
                "reject_type": "reject_persona",
                "evidence_code": p.get("evidence_code") or chk.get("curriculum_basis"),
                "trigger_term": p.get("trigger_term") or chk.get("trigger_term"),
                "verifier_stage": p.get("verifier_stage"),
                "flip_persona_id": other["id"],
            })

    out_pairs = all_pairs + new_t2
    with open(args.output, "w", encoding="utf-8") as f:
        for p in out_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_t1 = sum(1 for p in out_pairs if p.get("pair_type") == "step_pair")
    n_t2 = sum(1 for p in out_pairs if p.get("pair_type") == "belief_flip_pair")
    print(f"[done] 신규 Type-2 +{len(new_t2)} → 총 {len(out_pairs)} (Type-1 {n_t1}, Type-2 {n_t2}) → {args.output}")


if __name__ == "__main__":
    main()
