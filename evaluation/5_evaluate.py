"""
evaluation/5_evaluate.py

Step DPO / Full-Step DPO 공용 평가 스크립트.

평가 지표 (Table 1 기준):
1. final_answer_accuracy  — Final Acc.
2. step_accuracy          — Step Acc.    (acceptable step 비율)
3. persona_consistency    — Persona Cons. (1 - step_persona_err_rate)
4. belief_flip_win_rate   — Belief-Flip  (held-out Type-2 페어 logprob win rate)

Usage:
    python evaluation/5_evaluate.py \\
        --model checkpoints/bc_stepdpo \\
        --test-set data_pipeline/output/sft_test.jsonl \\
        --pairs data_pipeline/output/preference_pairs.jsonl \\
        --output checkpoints/bc_stepdpo/eval_results.json

--pairs: preference_pairs.jsonl (belief_flip_pair 포함). 생략 시 Belief-Flip 미측정.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402

# vLLM이 없는 환경(Mac M-series 등)에선 transformers fallback 사용.
try:
    from vllm import LLM, SamplingParams  # type: ignore  # noqa: E402
    _VLLM_AVAILABLE = True
except ImportError:
    from inference_backend import (  # noqa: E402
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )
    _VLLM_AVAILABLE = False

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from judge_prompts import (  # noqa: E402
    STEP_JUDGE_SYSTEM, STEP_JUDGE_USER_TEMPLATE, build_step_judge_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402
from openai_client import make_openai_client  # noqa: E402


def normalize_answer(s: str) -> str:
    return re.sub(r"[,\s]", "", s.strip().lower())


def extract_final_answer(text: str) -> str:
    m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text.lower())
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def generate(llm: LLM, problems: list[dict], personas: list[dict]) -> list[dict]:
    prompts, meta = [], []
    for p in problems:
        for pers in personas:
            prompts.append(f"{pers['tag']}\nProblem: {p['problem']}\nSolution:\n")
            meta.append({"problem": p, "persona": pers})
    sp = SamplingParams(temperature=0.0, max_tokens=800)
    outputs = llm.generate(prompts, sp)
    results = []
    for m, o in zip(meta, outputs):
        text = o.outputs[0].text
        results.append({
            "problem_id": m["problem"]["problem_id"],
            "problem": m["problem"]["problem"],
            "ground_truth": m["problem"]["ground_truth"],
            "persona": m["persona"],
            "solution_text": text,
            "steps": parse_steps(text),
            "predicted_answer": extract_final_answer(text),
        })
    return results


def metric_final_accuracy(results: list[dict]) -> float:
    correct = sum(
        normalize_answer(r["predicted_answer"]) == normalize_answer(r["ground_truth"])
        for r in results
    )
    return correct / max(1, len(results))


def metric_step_judge(client: OpenAI, results: list[dict]) -> dict:
    """Step-level: math accuracy + persona consistency 한꺼번에 측정."""
    n_accept, n_math_err, n_persona_err, n_total = 0, 0, 0, 0
    for r in results:
        if not r["steps"]:
            continue
        pers = r["persona"]
        sys_p = STEP_JUDGE_SYSTEM.format(**build_step_judge_kwargs(pers))
        user_p = STEP_JUDGE_USER_TEMPLATE.format(
            problem=r["problem"],
            ground_truth=r["ground_truth"],
            persona_tag=pers["tag"],
            solution_with_steps="\n".join(f"[{i+1}] {s}" for i, s in enumerate(r["steps"])),
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            out = json.loads(resp.choices[0].message.content)
            for s in out.get("steps", []):
                lbl = s.get("label")
                if lbl == "acceptable":
                    n_accept += 1
                elif lbl == "reject_math":
                    n_math_err += 1
                elif lbl == "reject_persona":
                    n_persona_err += 1
                n_total += 1
        except Exception:
            continue
    step_persona_err_rate = n_persona_err / max(1, n_total)
    return {
        "step_accuracy":       n_accept / max(1, n_total),
        "persona_consistency": 1.0 - step_persona_err_rate,
        "step_math_err_rate":  n_math_err / max(1, n_total),
        "step_persona_err_rate": step_persona_err_rate,
        "n_steps_judged": n_total,
    }


def _step_logprob(model, tokenizer, context: str, step: str, device: str) -> float:
    """teacher-forcing으로 step 토큰들의 log prob 합 반환."""
    ctx_ids  = tokenizer(context, add_special_tokens=False)["input_ids"]
    step_ids = tokenizer(step,    add_special_tokens=False)["input_ids"]
    if not step_ids:
        return 0.0
    full_ids = torch.tensor([ctx_ids + step_ids], dtype=torch.long, device=device)
    attn     = torch.ones_like(full_ids)
    with torch.no_grad():
        logits = model(input_ids=full_ids, attention_mask=attn).logits[0, :-1]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets   = full_ids[0, 1:]
    token_lp  = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # step 토큰 구간만 합산
    return token_lp[len(ctx_ids) - 1:].sum().item()


def metric_belief_flip(model_path: str, pairs_path: str, device: str = "cuda") -> dict:
    """held-out belief_flip_pair에서 logp(win) > logp(lose) 비율 측정.

    pairs_path: preference_pairs.jsonl (belief_flip_pair 포함)
    """
    flip_pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("pair_type") == "belief_flip_pair":
                flip_pairs.append(p)

    if not flip_pairs:
        return {"belief_flip_win_rate": None, "n_flip_pairs": 0}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    n_win = 0
    for p in flip_pairs:
        persona_tag  = p.get("persona_tag", "")
        prefix_steps = p.get("prefix_steps", [])
        context = (
            (f"{persona_tag}\n" if persona_tag else "")
            + f"Problem: {p['problem']}\nSolution:\n"
            + ("\n".join(prefix_steps) + "\n" if prefix_steps else "")
        )
        lp_win  = _step_logprob(model, tokenizer, context, p["step_win"],  device)
        lp_lose = _step_logprob(model, tokenizer, context, p["step_lose"], device)
        if lp_win > lp_lose:
            n_win += 1

    return {
        "belief_flip_win_rate": n_win / len(flip_pairs),
        "n_flip_pairs": len(flip_pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--pairs", default=None,
                        help="preference_pairs.jsonl — Belief-Flip 측정용 (생략 시 스킵)")
    parser.add_argument("--flip-stats", default=None,
                        help="flip_stats.json — 학습 데이터 flip 통계 (선택)")
    parser.add_argument("--personas-path", default="personas.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    client = make_openai_client()
    llm = LLM(model=args.model, dtype="bfloat16")
    personas = load_personas(args.personas_path)
    problems = [json.loads(l) for l in open(args.test_set, encoding="utf-8")]

    print(f"Generating {len(problems) * len(personas)} solutions...")
    results = generate(llm, problems, personas)

    print("Computing metrics...")
    final_acc    = metric_final_accuracy(results)
    step_metrics = metric_step_judge(client, results)

    metrics = {
        "final_answer_accuracy": final_acc,
        **step_metrics,
    }

    # Belief-Flip win rate
    if args.pairs and Path(args.pairs).exists():
        print("Computing Belief-Flip win rate...")
        flip_metrics = metric_belief_flip(args.model, args.pairs, device=args.device)
        metrics.update(flip_metrics)
    else:
        metrics["belief_flip_win_rate"] = None
        metrics["n_flip_pairs"] = 0

    if args.flip_stats and Path(args.flip_stats).exists():
        with open(args.flip_stats) as f:
            metrics["training_data_flip_stats"] = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "n_results": len(results)},
                  f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("Evaluation Results  [Table 1]")
    print("=" * 50)
    table_cols = [
        ("Final Acc.     ", "final_answer_accuracy"),
        ("Step Acc.      ", "step_accuracy"),
        ("Persona Cons.  ", "persona_consistency"),
        ("Belief-Flip    ", "belief_flip_win_rate"),
    ]
    for label, key in table_cols:
        v = metrics.get(key)
        print(f"  {label}: {f'{v:.4f}' if isinstance(v, float) else v}")
    print("-" * 50)
    print(f"  step_math_err_rate   : {metrics.get('step_math_err_rate', 0):.4f}")
    print(f"  n_steps_judged       : {metrics.get('n_steps_judged', 0)}")
    print(f"  n_flip_pairs         : {metrics.get('n_flip_pairs', 0)}")
    print(f"→ Full results in {args.output}")


if __name__ == "__main__":
    main()
