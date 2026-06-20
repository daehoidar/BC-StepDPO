"""data_pipeline_fullstepdpo/3a_mc_rollout_label.py

Full-Step DPO Stage 3a: 각 스텝의 두 채널 라벨 산출.

  step_value         := MC rollout 정답 도달률
  persona_validity   := PersonaVerifier 결과 (reject_persona → 0, else 1)

두 가지 운용 모드:
  (1) Shared 모드 (권장):
       --samples-path data_pipeline/output/samples_with_persona_labels.jsonl
       → π_ref 샘플링 + persona cascade SKIP (캐시 재사용)
       → MC rollout만 새로 호출 (PRM 라벨 산출은 본 모드 고유 작업)

  (2) Standalone 모드 (기존):
       --ref-model 만 주면 π_ref로 자체 샘플링 + cascade + MC rollout.

출력 한 행 = (problem, persona, sample, step):
  {
    "problem_id", "problem", "ground_truth", "persona_id", "persona_tag",
    "prefix_until_step":  [...],
    "step_idx": 2,
    "step_value": 0.625,
    "persona_validity": 1.0,
    "verifier_stage": "A"|"B"|"C",
    "evidence_code": ..., "trigger_term": ...,
    "sample_idx": 3
  }
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402
from openai_client import make_openai_client  # noqa: E402

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError:
    from inference_backend import (  # type: ignore
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )

from utils import parse_steps, load_personas  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


ANS_RE = re.compile(r"(?:answer|Answer)\s*(?:is)?\s*[:=]?\s*([\-\d\./]+)")


def extract_answer(text: str) -> str:
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def answer_matches(pred: str, gt: str) -> bool:
    p, g = pred.strip().lower(), gt.strip().lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return g in p or p in g


def build_prompt(persona_tag: str, problem: str, prefix_steps: list[str]) -> str:
    prefix_text = "\n".join(prefix_steps)
    if prefix_text:
        prefix_text += "\n"
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    return f"{persona_prefix}Problem: {problem}\nSolution:\n{prefix_text}"


def mc_step_value(
    llm, persona_tag: str, problem: str, prefix_steps: list[str], gt: str,
    m_rollouts: int, temperature: float = 1.0,
) -> float:
    prompt = build_prompt(persona_tag, problem, prefix_steps)
    sp = SamplingParams(temperature=temperature, max_tokens=400,
                        n=m_rollouts, stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    n_hit = 0
    for o in outputs[0].outputs:
        if answer_matches(extract_answer(o.text), gt):
            n_hit += 1
    return n_hit / max(1, m_rollouts)


def sample_chain(
    llm, persona_tag: str, problem: str, k: int, temperature: float = 0.9,
) -> list[list[str]]:
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


# ──────────────────────────── Shared 모드 ────────────────────────────────

def load_shared_samples(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def shared_mode(args, llm, fout):
    shared = load_shared_samples(Path(args.samples_path))
    print(f"[shared] {len(shared)} samples loaded")

    n_rows = 0
    for i, s in enumerate(shared):
        steps = s.get("steps", [])
        labels = s.get("step_persona_labels", [])
        if len(steps) < 2 or len(labels) != len(steps):
            continue
        limit = min(len(steps), args.max_steps_per_chain)
        persona_tag = s.get("persona_tag", "")
        for t in range(1, limit + 1):
            prefix = steps[:t]
            v_math = mc_step_value(
                llm, persona_tag, s["problem"], prefix,
                s["ground_truth"], m_rollouts=args.m_rollouts,
            )
            p_lab = labels[t - 1]
            v_persona = 0.0 if p_lab.get("verdict") == "reject_persona" else 1.0
            fout.write(json.dumps({
                "problem_id": s["problem_id"],
                "problem": s["problem"],
                "ground_truth": s["ground_truth"],
                "persona_id": s["persona_id"],
                "persona_tag": persona_tag,
                "prefix_until_step": prefix,
                "step_idx": t,
                "step_value": v_math,
                "persona_validity": v_persona,
                "verifier_stage": p_lab.get("stage"),
                "evidence_code": p_lab.get("evidence_code"),
                "trigger_term": p_lab.get("trigger_term"),
                "sample_idx": s.get("sample_idx", -1),
            }, ensure_ascii=False) + "\n")
            n_rows += 1
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(shared)}] step-rows: {n_rows}")
    print(f"Done (shared). {n_rows} step rows.")


# ──────────────────────────── Standalone 모드 ────────────────────────────

def standalone_mode(args, llm, fout):
    gpt_client = make_openai_client()
    stage_b_client = None
    if not args.disable_stage_b:
        stage_b_client = OpenAI(base_url=args.verifier_base_url,
                                api_key=args.verifier_api_key)
    verifier = PersonaVerifier(
        stage_b_client=stage_b_client,
        stage_b_model=args.verifier_model,
        stage_c_client=gpt_client,
        stage_c_model=args.gpt_model,
        stage_b_conf_threshold=args.stage_b_threshold,
        enable_stage_b=not args.disable_stage_b,
        enable_stage_c=not args.disable_stage_c,
        stage_log_path=args.stage_log_path or None,
    )
    if args.stage_log_path:
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}

    rows = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[load] {len(rows)} (problem, persona) rows")

    n_rows = 0
    for i, prob in enumerate(rows):
        persona = persona_by_id.get(prob.get("persona", ""))
        if persona is None:
            continue
        persona_tag = persona.get("tag", "")
        verifier.problem_context = f"{prob.get('problem_id','?')}::{persona['id']}"

        chains = sample_chain(llm, persona_tag, prob["question"],
                              k=args.k_samples)
        for sample_idx, steps in enumerate(chains):
            if len(steps) < 2:
                continue
            limit = min(len(steps), args.max_steps_per_chain)
            for t in range(1, limit + 1):
                prefix = steps[:t]
                v_math = mc_step_value(
                    llm, persona_tag, prob["question"], prefix,
                    prob["gt_answer"], m_rollouts=args.m_rollouts,
                )
                p_res = verifier.verify_step(
                    steps[t - 1], persona, prefix=steps[: t - 1],
                )
                v_persona = 0.0 if p_res.verdict == "reject_persona" else 1.0
                fout.write(json.dumps({
                    "problem_id": prob["problem_id"],
                    "problem": prob["question"],
                    "ground_truth": prob["gt_answer"],
                    "persona_id": persona["id"],
                    "persona_tag": persona_tag,
                    "prefix_until_step": prefix,
                    "step_idx": t,
                    "step_value": v_math,
                    "persona_validity": v_persona,
                    "verifier_stage": p_res.stage,
                    "evidence_code": p_res.evidence_code,
                    "trigger_term": p_res.trigger_term,
                    "sample_idx": sample_idx,
                }, ensure_ascii=False) + "\n")
                n_rows += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(rows)}] step-rows: {n_rows} "
                  f"cascade: {verifier.dump_counters()}")
    print(f"Done (standalone). {n_rows} step rows.")
    print(f"Final cascade counters: {verifier.dump_counters()}")


def main():
    ap = argparse.ArgumentParser()
    # 공통
    ap.add_argument("--ref-model", required=True,
                    help="π_ref 경로. shared 모드에서도 MC rollout에 필요.")
    ap.add_argument("--m-rollouts", type=int, default=8)
    ap.add_argument("--max-steps-per-chain", type=int, default=10)
    ap.add_argument("--output",
                    default="data_pipeline_fullstepdpo/output/step_values.jsonl")

    # Shared 모드
    ap.add_argument("--samples-path", default=None,
                    help="shared_sampling.py 산출 jsonl 경로")

    # Standalone 모드
    ap.add_argument("--seed-problems", default=None,
                    help="standalone 모드용 (problem × persona) jsonl")
    ap.add_argument("--personas-path",
                    default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--k-samples", type=int, default=6)
    ap.add_argument("--verifier-base-url", default="http://localhost:8001/v1")
    ap.add_argument("--verifier-model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--verifier-api-key", default="EMPTY")
    ap.add_argument("--stage-b-threshold", type=float, default=0.85)
    ap.add_argument("--disable-stage-b", action="store_true")
    ap.add_argument("--disable-stage-c", action="store_true")
    ap.add_argument("--gpt-model", default="gpt-4o")
    ap.add_argument("--stage-log-path",
                    default="data_pipeline_fullstepdpo/output/stage_log.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # PRM 모드는 ref-model이 MC rollout에 항상 필요
    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)

    with open(out_path, "w", encoding="utf-8") as fout:
        if args.samples_path:
            print("[mode] SHARED — reusing samples + persona labels, "
                  "MC rollout new")
            shared_mode(args, llm, fout)
        else:
            print("[mode] STANDALONE — sampling + cascade + MC rollout")
            if not args.seed_problems:
                raise SystemExit(
                    "Standalone 모드는 --seed-problems 필요."
                    " 또는 --samples-path 로 shared 산출물을 넘기세요."
                )
            standalone_mode(args, llm, fout)


if __name__ == "__main__":
    main()
