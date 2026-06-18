"""rejudge_contaminated.py — StageC judge 실패로 강제 persona_ok 처리된 오염 라벨 재판정.

팀 키 rate-limit(429)·hard-timeout 시 verify_step이 fallback으로 persona_ok를
기록한다(reasoning에 "stage-C error"/"verify failed"/"429"/"Error code" 흔적).
이 스크립트는 그런 step만 골라 PersonaVerifier(StageC)로 다시 판정해 라벨을 교체한다.

- samples_with_persona_labels.jsonl 전체를 읽어, 오염 라벨이 있는 step만 재판정.
- 재판정은 thread pool로 병렬(개인 키 빠름). prefix(steps[:j])를 그대로 복원해 문맥 동일.
- 임시파일에 쓰고 원자적 rename → 부분 기록 손상 방지.

사용:
  python data_pipeline/rejudge_contaminated.py --samples-path <jsonl> \
      --personas-path personas.json --gpt-model gpt-4o-mini
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import load_personas  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402

CONTAM_MARKERS = ("stage-C error", "verify failed", "429", "Error code")


def is_contaminated(label: dict) -> bool:
    r = (label.get("reasoning") or "")
    return any(m in r for m in CONTAM_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-path", required=True)
    ap.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.samples_path)
    if not path.exists():
        print(f"[rejudge] {path} 없음 — skip")
        return

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}
    verifier = PersonaVerifier(
        stage_c_client=make_openai_client(), stage_c_model=args.gpt_model,
        enable_stage_b=False, enable_stage_c=True, stage_log_path=None,
    )

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    # 재판정 대상 수집: (row_idx, step_j, step_text, prefix, persona)
    tasks = []
    for ri, r in enumerate(rows):
        persona = persona_by_id.get(r.get("persona_id"))
        if persona is None:
            continue
        steps = r.get("steps", [])
        labels = r.get("step_persona_labels", [])
        if len(labels) != len(steps):
            continue
        for j, lab in enumerate(labels):
            if is_contaminated(lab):
                tasks.append((ri, j, steps[j], steps[:j], persona))

    print(f"[rejudge] 오염 step {len(tasks)}개 재판정 시작 (workers={args.workers})")
    if not tasks:
        print("[rejudge] 재판정 대상 없음 — 종료")
        return

    def _do(t):
        ri, j, step, prefix, persona = t
        res = verifier.verify_step(step, persona, prefix=prefix)
        return ri, j, {
            "verdict": res.verdict, "confidence": res.confidence,
            "stage": res.stage, "trigger_term": res.trigger_term,
            "evidence_code": res.evidence_code,
            "first_introduced": res.first_introduced, "reasoning": res.reasoning,
        }

    flipped = 0; still_err = 0; done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ri, j, new_lab in ex.map(_do, tasks):
            old = rows[ri]["step_persona_labels"][j]["verdict"]
            rows[ri]["step_persona_labels"][j] = new_lab
            done += 1
            if is_contaminated(new_lab):
                still_err += 1
            elif new_lab["verdict"] != old:
                flipped += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(tasks)} (flip {flipped}, 여전히오류 {still_err})")

    tmp = path.with_suffix(path.suffix + ".rejudged.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[rejudge] 완료: {done}개 재판정, persona_ok→reject 전환 {flipped}개, "
          f"여전히 API오류 {still_err}개. → {path}")


if __name__ == "__main__":
    main()
