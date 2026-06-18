"""sft_vs_trained.py — SFT(π_ref) vs BC-StepDPO 학습후 모델의 생성 비교.

같은 (문제 × 페르소나)에 대해:
  - SFT:    merged SFT 모델 단독
  - trained: merged SFT + bc_stepdpo_v2 LoRA
greedy(seed 고정)로 생성해 나란히 출력 → 과적합으로 망가졌는지/수준 적합성이 좋아졌는지 정성 확인.
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
from utils import load_personas  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="checkpoints/sft_qwen3_1.7b_eos_merged")
    ap.add_argument("--adapter", default="checkpoints/bc_stepdpo_v2")
    ap.add_argument("--seed-problems", default="data_pipeline/output/seed_problems.jsonl")
    ap.add_argument("--n-problems", type=int, default=2)
    ap.add_argument("--personas", nargs="+",
                    default=["elem_low", "elem_high", "high_low", "high_high"])
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 문제 N개 (고정 순서)
    rows = [json.loads(l) for l in open(args.seed_problems)]
    seen, probs = set(), []
    for r in rows:
        q = r["question"]
        if q in seen:
            continue
        seen.add(q); probs.append(r)
        if len(probs) >= args.n_problems:
            break

    personas = {p["id"]: p for p in load_personas(REPO_ROOT / "personas.json")}
    tok = AutoTokenizer.from_pretrained(args.merged)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_kwargs = dict(max_new_tokens=args.max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                      repetition_penalty=1.2, no_repeat_ngram_size=6)

    def gen(model, prompt):
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("=== load merged SFT ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=dtype).to(device).eval()

    results = []
    for prob in probs:
        for pid in args.personas:
            persona = personas[pid]
            prompt = f"{persona['tag']}\nProblem: {prob['question']}\nSolution:\n"
            results.append([prob["question"], pid, persona.get("grade_band"), gen(base, prompt), None])

    print("=== load + apply bc_stepdpo_v2 LoRA ===", flush=True)
    trained = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    idx = 0
    for prob in probs:
        for pid in args.personas:
            persona = personas[pid]
            prompt = f"{persona['tag']}\nProblem: {prob['question']}\nSolution:\n"
            results[idx][4] = gen(trained, prompt); idx += 1

    for q, pid, gb, sft_out, tr_out in results:
        print("\n" + "#" * 90)
        print(f"PROBLEM: {q[:120]}")
        print(f"PERSONA: {pid} ({gb})")
        print("-" * 90)
        print(f"[SFT]\n{sft_out[:600]}")
        print("-" * 90)
        print(f"[+BC-StepDPO]\n{tr_out[:600]}")


if __name__ == "__main__":
    main()
