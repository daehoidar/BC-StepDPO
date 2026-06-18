"""eval_belief_flip.py — Belief-Flip Accuracy 지표 (실험용).

정의:
  각 테스트 문제에 대해 모델이 저수준 페르소나 b_lo(초등)와 고수준 b_hi(고교)로
  풀이를 각각 생성한다. 교육과정 근거 persona judge(StageC)로 평가하여,
    (i)   sol_lo 가 b_lo 에 적합(persona_ok),
    (ii)  sol_hi 가 b_hi 에 적합,
    (iii) sol_hi 가 b_lo 에는 부적합(reject_persona) — 모델이 수준에 맞게 실제로
          표현을 차별화(=belief-flip)했음.
  세 조건을 모두 만족한 문제를 '올바른 flip'으로 센다.
  Belief-Flip Accuracy = (올바른 flip 문제 수) / (전체 문제 수).

보조 지표:
  adapt_rate  = (iii)만의 비율 (모델이 b_hi 풀이를 b_lo엔 부적합하게 만든 비율)
  lo_ok_rate  = (i),  hi_ok_rate = (ii)

생성은 transformers(PeftModel)로 — base(merged SFT) + LoRA adapter. adapter 없으면 SFT baseline.
판정은 PersonaVerifier StageC(gpt-4o-mini).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils import load_personas, parse_steps  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


def persona_ok(verifier, steps, persona) -> bool:
    """모든 step이 persona_ok면 적합(reject_persona가 하나라도 있으면 부적합)."""
    for j, s in enumerate(steps):
        r = verifier.verify_step(s, persona, prefix=steps[:j])
        if r.verdict == "reject_persona":
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="checkpoints/sft_qwen3_1.7b_eos_merged")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 경로 (없으면 SFT baseline)")
    ap.add_argument("--test-set", default="data_pipeline/output/sft_test_eval60.jsonl")
    ap.add_argument("--persona-low", default="elem_low")
    ap.add_argument("--persona-high", default="high_high")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--gpt-model", default="gpt-4o-mini")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    personas = {p["id"]: p for p in load_personas(REPO_ROOT / "personas.json")}
    p_lo, p_hi = personas[args.persona_low], personas[args.persona_high]

    # 고유 문제 N개
    seen, probs = set(), []
    for line in open(args.test_set, encoding="utf-8"):
        r = json.loads(line)
        q = r.get("problem") or r.get("question")
        if not q or q in seen:
            continue
        seen.add(q); probs.append(q)
        if len(probs) >= args.n_problems:
            break

    tok = AutoTokenizer.from_pretrained(args.merged)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=dtype).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device).eval()

    gen_kw = dict(max_new_tokens=args.max_new, do_sample=False,
                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                  repetition_penalty=1.2, no_repeat_ngram_size=6)

    def gen(persona, q):
        prompt = f"{persona['tag']}\nProblem: {q}\nSolution:\n"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
        txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_steps(txt)

    verifier = PersonaVerifier(stage_c_client=make_openai_client(), stage_c_model=args.gpt_model,
                              enable_stage_b=False, enable_stage_c=True, stage_log_path=None)

    n = 0; n_lo_ok = 0; n_hi_ok = 0; n_adapt = 0; n_flip = 0
    for q in probs:
        sol_lo = gen(p_lo, q)
        sol_hi = gen(p_hi, q)
        if len(sol_lo) < 1 or len(sol_hi) < 1:
            continue
        n += 1
        i = persona_ok(verifier, sol_lo, p_lo)           # (i) sol_lo OK for low
        ii = persona_ok(verifier, sol_hi, p_hi)          # (ii) sol_hi OK for high
        iii = not persona_ok(verifier, sol_hi, p_lo)     # (iii) sol_hi NOT OK for low (차별화)
        n_lo_ok += i; n_hi_ok += ii; n_adapt += iii
        if i and ii and iii:
            n_flip += 1
        print(f"[{n}] lo_ok={i} hi_ok={ii} adapt={iii}  flip={i and ii and iii}", flush=True)

    res = {
        "model": args.adapter or args.merged,
        "persona_low": args.persona_low, "persona_high": args.persona_high,
        "n_problems": n,
        "belief_flip_accuracy": round(100 * n_flip / max(n, 1), 1),
        "lo_ok_rate": round(100 * n_lo_ok / max(n, 1), 1),
        "hi_ok_rate": round(100 * n_hi_ok / max(n, 1), 1),
        "adapt_rate": round(100 * n_adapt / max(n, 1), 1),
    }
    print("\n=== Belief-Flip ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.output, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
