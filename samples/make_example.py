"""samples/make_example.py
실제 GPT-4o 호출로 walkthrough 예시 데이터를 한 문제·6 페르소나에 대해 생성한다.

데이터 파이프라인이 완전 구현된 상태를 가정하고, 한 문제에 대한 *모든* 산출
파일을 만든다. SFT 모델이 아직 학습되지 않았으므로 [3_collect_errors.sh] 단계는
**gpt-4o-mini를 SFT 모델 proxy**로 사용하여 시뮬레이션한다. 실제 SFT 모델이
준비되면 SFT_PROXY_MODEL만 갈아끼우면 된다.

생성되는 파일:
  1. seed_problems.jsonl     — 6 페르소나로 펼친 시드 (호출 없음)
  2. sft_data.jsonl          — 페르소나별 정답 풀이 (GPT-4o × 6)
  3. predictions.jsonl       — SFT proxy N샘플링 결과 (gpt-4o-mini × 48)
  4. step_pairs.jsonl        — Type-1: first-error localize + rectify로 빌드
                               (GPT-4o judge × ~24, GPT-4o rectify × ~6)
  5. alt_steps.jsonl         — 1c anchored continuation 원본 (GPT-4o × ~12)
                               + math judge 결과 (GPT-4o × ~12)
  6. belief_pairs.jsonl      — Type-2: Step 1 cross-persona + anchored (judge 통과분)
  7. train.jsonl             — step_pairs + belief_pairs 병합

비용 자릿수: 한 문제당 ~$0.50 (모델 가격 변동 가능).

사용:
    export OPENAI_API_KEY=sk-...
    python samples/make_example.py
    python samples/make_example.py --skip-sft           # SFT 합성 건너뛰기
    python samples/make_example.py --skip-type1         # Type-1 건너뛰기 (Type-2만)
    python samples/make_example.py --n-samples 4        # SFT proxy 샘플 수 조정
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personas import all_personas, render_system_prompt  # noqa: E402

# 모델 -----------------------------------------------------------------------
SFT_MODEL = "gpt-4o"                # SFT 데이터 합성 (1번 단계)
# SFT proxy: 미래 SFT base model 시뮬레이션.
# 실제 SFT base는 **Qwen3-4B** (Instruct), 학습 후 outputs/pi_sft_qwen3_4b/ 사용 예정.
# 로컬 Ollama로 갈아끼우려면 `SFT_PROXY_MODEL="qwen3:4b"` + base_url 분리 client 추가.
SFT_PROXY_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"              # math judge + first-error localize + rectify

SFT_TEMPERATURE = 0.3
SFT_PROXY_TEMPERATURE = 1.0         # 다양성 ↑ 실패 유도
ALT_TEMPERATURE = 0.4
JUDGE_TEMPERATURE = 0.0
RECTIFY_TEMPERATURE = 0.3

MAX_OUTPUT_TOKENS = 1024
ALT_MAX_TOKENS = 256
JUDGE_MAX_TOKENS = 64
RECTIFY_MAX_TOKENS = 256

# 기본값 ---------------------------------------------------------------------
# MetaMathQA(영어) 분포 대표하는 GSM-style 문제 1개. 사용자 --question으로 갈아끼우면 됨.
DEFAULT_QUESTION = (
    "James buys 5 packs of pencils. Each pack contains 12 pencils. "
    "He uses 4 pencils per day for schoolwork. "
    "For how many days will the pencils last?"
)
DEFAULT_GT = "15"
DEFAULT_PROBLEM_ID = "demo_pencils_001"
DEFAULT_N_SAMPLES = 8

# 1c anchored continuation용 instruction (English).
ANCHORED_INSTRUCTION = """Additional instructions (override the [Output Format] section above):
- You are continuing a partial solution another student already started. Write ONLY the next step.
- The provided prefix is a FIXED input. Do NOT modify, rewrite, or omit any part of it.
- Do NOT match your tone to the prefix's tone. Keep YOUR persona's tone and explanation depth.
- Output exactly one step and stop immediately. No evaluation, summary, additional steps, or \\boxed.
- Write only steps you are mathematically confident in. If unsure, keep it short."""

_STEP_HEADER_RX = re.compile(r"^Step\s+\d+:")


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def call_model(client, model, system_prompt, user_msg, temperature, max_tokens, seed=None):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def split_steps(solution: str) -> list[str]:
    """'preamble\\nStep 1: ...' -> ['Step 1: ...', 'Step 2: ...']

    Step 헤더로 시작하지 않는 chunk(GPT가 풀이 앞에 붙이는 prelude)는 drop.
    """
    parts = re.split(r"(?=Step\s+\d+:)", solution.strip())
    return [p.strip() for p in parts if _STEP_HEADER_RX.match(p.strip())]


def step_n_only(solution: str, n: int) -> str | None:
    steps = split_steps(solution)
    if 0 < n <= len(steps):
        return steps[n - 1]
    return None


def first_n_steps_text(solution: str, n: int) -> str:
    steps = split_steps(solution)
    return "\n".join(steps[:n])


def build_prompt(persona_id: str, question: str) -> str:
    """1_synthesize_sft.py의 user_msg 형식과 동일하게 (English)."""
    return f"<{persona_id}>\n{question}\n\nPlease solve the problem step by step."


def extract_boxed(text: str) -> str | None:
    """\\boxed{...} 안의 내용 추출. 가장 마지막 \\boxed만."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str | None) -> str | None:
    """'5/6', '\\frac{5}{6}', ' 5/6 ' 등을 비교 가능한 형태로 통일."""
    if ans is None:
        return None
    ans = re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", ans)
    ans = re.sub(r"\\dfrac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.strip(".$")
    return ans or None


def answer_correct(solution: str, gt_answer: str) -> bool:
    return normalize_answer(extract_boxed(solution)) == normalize_answer(gt_answer)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 1. SFT 합성 (1번 단계)
# ---------------------------------------------------------------------------
def synthesize_sft(client, personas, question, problem_id, gt_answer):
    rows = []
    for p in personas:
        sys_p = render_system_prompt(p)
        user_msg = build_prompt(p["id"], question)
        sol = call_model(client, SFT_MODEL, sys_p, user_msg,
                         SFT_TEMPERATURE, MAX_OUTPUT_TOKENS)
        correct = answer_correct(sol, gt_answer)
        rows.append({
            "problem_id": problem_id,
            "persona": p["id"],
            "question": question,
            "gt_answer": gt_answer,
            "difficulty": "easy",
            "gpt4o_solution": sol,
            "model": SFT_MODEL,
            "answer_correct": correct,
        })
        n_steps = len(split_steps(sol))
        flag = "✓" if correct else "✗"
        print(f"  [sft] {p['id']}: {n_steps} steps, ans {flag}")
    return rows


# ---------------------------------------------------------------------------
# 2. Math judge (1d 단계 + 4번 단계 공용 단발 판정)
# ---------------------------------------------------------------------------
MATH_JUDGE_PROMPT = """You are a judge that checks ONLY mathematical correctness. Do NOT evaluate persona tone or style.

[Problem]
{question}

[Solution prefix so far]
{prefix}

[Step under review]
{step}

Decide whether the step above is a mathematically valid continuation of the prefix.
Reply with exactly one word: 'pass' if correct, 'fail' if not."""


def math_judge(client, question, prefix, step):
    sys_p = "You are a math-correctness judge. Do not evaluate tone."
    user_msg = MATH_JUDGE_PROMPT.format(
        question=question, prefix=prefix or "(none)", step=step
    )
    resp = call_model(client, JUDGE_MODEL, sys_p, user_msg,
                      JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
    return "pass" if "pass" in resp.lower() else "fail"


# ---------------------------------------------------------------------------
# 3. Type-2 Step 1 cross-persona pairs (free + sft answer_correct 기반)
# ---------------------------------------------------------------------------
def build_belief_step1_pairs(sft_rows):
    pairs = []
    for i, target in enumerate(sft_rows):
        if not target.get("answer_correct", True):
            continue
        t_step1 = step_n_only(target["gpt4o_solution"], 1)
        if not t_step1:
            continue
        for j, alt in enumerate(sft_rows):
            if i == j or not alt.get("answer_correct", True):
                continue
            a_step1 = step_n_only(alt["gpt4o_solution"], 1)
            if not a_step1:
                continue
            pairs.append({
                "type": "belief_pair_step1",
                "persona": target["persona"],
                "alt_persona": alt["persona"],
                "problem_id": target["problem_id"],
                "prompt": build_prompt(target["persona"], target["question"]),
                "prefix": "",
                "chosen": t_step1,
                "rejected": a_step1,
                "math_status": "both_pass_by_final_answer",
            })
    return pairs


# ---------------------------------------------------------------------------
# 4. Type-2 anchored continuation (1c 단계) + math judge
# ---------------------------------------------------------------------------
def anchored_call(client, persona, prefix_text, question, cut_k):
    sys_p = render_system_prompt(persona) + "\n\n" + ANCHORED_INSTRUCTION
    user_msg = (
        f"<problem>\n{question}\n</problem>\n\n"
        f"<solution_so_far>\n{prefix_text}\n</solution_so_far>\n\n"
        f"Write only the next single step that follows the prefix above. Format:\n"
        f"Step {cut_k}: <one short line or short paragraph>"
    )
    resp = call_model(client, SFT_MODEL, sys_p, user_msg,
                      ALT_TEMPERATURE, ALT_MAX_TOKENS)
    m = re.search(rf"Step\s+{cut_k}:\s*(.+?)(?=\n\s*Step\s+\d+:|\Z)", resp, re.DOTALL)
    body = m.group(1).strip() if m else resp.strip().split("\nStep ")[0].strip()
    return f"Step {cut_k}: {body}"


def synthesize_anchored_pairs(client, sft_rows, persona_by_id, cut_k, question):
    pairs = []
    alt_step_rows = []
    persona_ids = [r["persona"] for r in sft_rows]
    for i, target in enumerate(sft_rows):
        t_persona = persona_by_id[target["persona"]]
        prefix_text = first_n_steps_text(target["gpt4o_solution"], cut_k - 1)
        if not prefix_text:
            print(f"  [anchored] {target['persona']}: prefix 부족 (cut_k={cut_k}), skip")
            continue
        alt_id = persona_ids[(i + 3) % len(persona_ids)]
        if alt_id == target["persona"]:
            alt_id = persona_ids[(i + 1) % len(persona_ids)]
        alt_persona = persona_by_id[alt_id]

        chosen_regen = anchored_call(client, t_persona, prefix_text, question, cut_k)
        alt_step = anchored_call(client, alt_persona, prefix_text, question, cut_k)
        # math judge 둘 다
        chosen_judge = math_judge(client, question, prefix_text, chosen_regen)
        alt_judge = math_judge(client, question, prefix_text, alt_step)

        header = f"Step {cut_k}:"
        chosen_body = " " + chosen_regen.removeprefix(header).lstrip()
        rejected_body = " " + alt_step.removeprefix(header).lstrip()

        passed = (chosen_judge == "pass" and alt_judge == "pass")
        flag = "✓ both pass" if passed else f"✗ chosen={chosen_judge}, alt={alt_judge}"
        print(f"  [anchored] target={target['persona']}, alt={alt_id}: {flag}")

        alt_step_rows.append({
            "problem_id": target["problem_id"],
            "persona": target["persona"],
            "alt_persona": alt_id,
            "prefix": prefix_text + "\n",
            "cut_point": cut_k,
            "chosen_regen": chosen_regen,
            "alt_step": alt_step,
            "chosen_math_judge": chosen_judge,
            "alt_math_judge": alt_judge,
            "model": SFT_MODEL,
            "temperature": ALT_TEMPERATURE,
        })
        if passed:
            pairs.append({
                "type": "belief_pair_anchored",
                "persona": target["persona"],
                "alt_persona": alt_id,
                "problem_id": target["problem_id"],
                "prompt": build_prompt(target["persona"], target["question"]),
                "prefix": prefix_text + "\n" + header,
                "chosen": chosen_body,
                "rejected": rejected_body,
                "cut_point": cut_k,
                "math_status": "both_pass",
            })
    return pairs, alt_step_rows


# ---------------------------------------------------------------------------
# 5. SFT proxy 샘플링 (3번 단계 시뮬레이션)
# ---------------------------------------------------------------------------
def sft_proxy_sampling(client, personas, question, gt_answer, problem_id, n_samples):
    """gpt-4o-mini로 페르소나 조건부 풀이 N번 샘플링. \\boxed로 정/오 라벨링."""
    all_rows = []
    for p in personas:
        sys_p = render_system_prompt(p)
        user_msg = build_prompt(p["id"], question)
        samples = []
        for k in range(n_samples):
            sol = call_model(client, SFT_PROXY_MODEL, sys_p, user_msg,
                             SFT_PROXY_TEMPERATURE, MAX_OUTPUT_TOKENS,
                             seed=1000 + k)
            samples.append({
                "attempt_index": k,
                "solution": sol,
                "result": answer_correct(sol, gt_answer),
            })
        n_fail = sum(1 for s in samples if not s["result"])
        rate = n_fail / n_samples
        print(f"  [sft-proxy] {p['id']}: {n_samples - n_fail}/{n_samples} 정답, failure_rate={rate:.2f}")
        all_rows.append({
            "problem_id": problem_id,
            "persona": p["id"],
            "question": question,
            "gt_answer": gt_answer,
            "n_samples": n_samples,
            "n_failures": n_fail,
            "failure_rate": rate,
            "sampling_config": {
                "model": SFT_PROXY_MODEL,
                "temperature": SFT_PROXY_TEMPERATURE,
                "top_p": 1.0,
                "max_tokens": MAX_OUTPUT_TOKENS,
            },
            "samples": samples,
        })
    return all_rows


# ---------------------------------------------------------------------------
# 6. Locate first error + rectify (4번 + 6번 단계)
# ---------------------------------------------------------------------------
LOCATE_PROMPT = """The following is a student's math solution. The correct answer is {gt_answer}, but the student's final answer is wrong.

[Problem]
{question}

[Student solution]
{solution}

Review each step from the beginning and identify the **step number where the first mathematical error occurs** (1-indexed).
- Do NOT evaluate persona tone or style. Only check mathematical correctness.
- If no step has an obvious local error but the final answer is still wrong, return the last step number.

Reply in EXACTLY this one-line format:
First error step: <number>"""


def locate_first_error(client, question, gt_answer, solution):
    sys_p = "You are a judge that locates the first mathematical error in a step-by-step solution."
    user_msg = LOCATE_PROMPT.format(
        gt_answer=gt_answer, question=question, solution=solution
    )
    resp = call_model(client, JUDGE_MODEL, sys_p, user_msg,
                      JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
    m = re.search(r"First\s*error\s*step:\s*(\d+)", resp, re.IGNORECASE)
    return int(m.group(1)) if m else None


RECTIFY_PROMPT = """The prefix below is correct up through Step {k_minus_1}. Starting from Step {k} the solution went wrong.
Maintaining your persona's tone, **rewrite ONLY Step {k} so it is mathematically correct**.

[Problem]
{question}
[Correct final answer]
{gt_answer}

[Prefix (Step 1 ~ Step {k_minus_1})]
{prefix}

[Original wrong Step {k} (for reference only)]
{wrong}

Rules:
- Keep your persona's tone, vocabulary, and explanation depth.
- Output ONLY Step {k}, one step. Do not add any further steps. No \\boxed.

Format:
Step {k}: <correct step content>"""


def rectify_step(client, persona, question, gt_answer, prefix, wrong_step, k):
    sys_p = (render_system_prompt(persona)
             + "\n\nIgnore the [Output Format] section above and follow the rules in the user message instead.")
    user_msg = RECTIFY_PROMPT.format(
        k=k, k_minus_1=k - 1, question=question, gt_answer=gt_answer,
        prefix=prefix or "(none)", wrong=wrong_step,
    )
    resp = call_model(client, JUDGE_MODEL, sys_p, user_msg,
                      RECTIFY_TEMPERATURE, RECTIFY_MAX_TOKENS)
    m = re.search(rf"Step\s+{k}:\s*(.+?)(?=\n\s*Step\s+\d+:|\Z)", resp, re.DOTALL)
    body = m.group(1).strip() if m else resp.strip().split("\nStep ")[0].strip()
    return f"Step {k}: {body}"


def build_type1_step_pairs(client, prediction_rows, persona_by_id, question, gt_answer):
    """각 persona에서 첫 실패 샘플을 골라 locate→rectify→step_pair 생성."""
    pairs = []
    for pred in prediction_rows:
        if pred["n_failures"] == 0:
            print(f"  [type1] {pred['persona']}: 실패 0, step_pair 생성 안 함")
            continue
        failure = next(s for s in pred["samples"] if not s["result"])
        wrong_sol = failure["solution"]
        steps = split_steps(wrong_sol)
        if not steps:
            print(f"  [type1] {pred['persona']}: Step 헤더 없음, skip")
            continue
        err_k = locate_first_error(client, question, gt_answer, wrong_sol)
        if err_k is None or err_k < 1 or err_k > len(steps):
            print(f"  [type1] {pred['persona']}: locate 실패({err_k}), skip")
            continue
        prefix = "\n".join(steps[:err_k - 1])
        wrong_step_full = steps[err_k - 1]  # "Step k: ..." 전체
        header = f"Step {err_k}:"
        rejected_body = " " + wrong_step_full.removeprefix(header).lstrip()
        chosen_full = rectify_step(
            client, persona_by_id[pred["persona"]],
            question, gt_answer, prefix, wrong_step_full, err_k,
        )
        chosen_body = " " + chosen_full.removeprefix(header).lstrip()
        # rectify 결과를 judge로 한 번 더 확인
        rectify_judge = math_judge(client, question, prefix, chosen_full)
        if rectify_judge != "pass":
            print(f"  [type1] {pred['persona']}: rectify judge 실패, skip")
            continue
        print(f"  [type1] {pred['persona']}: cut_point={err_k}, attempt={failure['attempt_index']}, ✓")
        pairs.append({
            "type": "step_pair",
            "persona": pred["persona"],
            "problem_id": pred["problem_id"],
            "prompt": build_prompt(pred["persona"], pred["question"]),
            "prefix": (prefix + "\n" if prefix else "") + header,
            "chosen": chosen_body,
            "rejected": rejected_body,
            "gt_answer": gt_answer,
            "cut_point": err_k,
            "n_samples": pred["n_samples"],
            "n_failures": pred["n_failures"],
            "attempt_index": failure["attempt_index"],
            "failure_rate": pred["failure_rate"],
            "sampling_config": pred["sampling_config"],
            "rectify_model": JUDGE_MODEL,
            "locate_model": JUDGE_MODEL,
        })
    return pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--gt-answer", default=DEFAULT_GT)
    ap.add_argument("--problem-id", default=DEFAULT_PROBLEM_ID)
    ap.add_argument("--cut-k", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                    help="SFT proxy 샘플 수 (3번 단계 rep)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "samples"))
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-type1", action="store_true",
                    help="Type-1 (SFT proxy 샘플링 + locate + rectify) 전체 건너뛰기")
    ap.add_argument("--skip-type2", action="store_true",
                    help="Type-2 (Step 1 + anchored) 건너뛰기")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("[error] OPENAI_API_KEY 환경변수를 먼저 설정하세요.")

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = all_personas()
    persona_by_id = {p["id"]: p for p in personas}

    # 0. seed_problems
    print("[0/6] seed_problems.jsonl")
    seed_rows = [{
        "problem_id": args.problem_id,
        "persona": p["id"],
        "question": args.question,
        "gt_answer": args.gt_answer,
        "difficulty": "easy",
    } for p in personas]
    write_jsonl(out_dir / "seed_problems.jsonl", seed_rows)

    # 1. SFT 합성
    sft_path = out_dir / "sft_data.jsonl"
    if args.skip_sft and sft_path.exists():
        print("[1/6] sft_data.jsonl (재사용)")
        with open(sft_path, encoding="utf-8") as f:
            sft_rows = [json.loads(l) for l in f]
    else:
        print(f"[1/6] sft_data.jsonl (GPT-4o × {len(personas)})")
        sft_rows = synthesize_sft(client, personas, args.question,
                                  args.problem_id, args.gt_answer)
        write_jsonl(sft_path, sft_rows)

    # 2-4. Type-2
    belief_pairs = []
    if not args.skip_type2:
        print("[2/6] belief_pair Step 1 (free)")
        step1_pairs = build_belief_step1_pairs(sft_rows)
        print(f"  built {len(step1_pairs)} cross-persona Step 1 pairs")

        print(f"[3/6] anchored continuation + math judge (GPT-4o × ~{4 * len(sft_rows)})")
        anchored_pairs, alt_step_rows = synthesize_anchored_pairs(
            client, sft_rows, persona_by_id, args.cut_k, args.question
        )
        write_jsonl(out_dir / "alt_steps.jsonl", alt_step_rows)
        belief_pairs = step1_pairs + anchored_pairs
        write_jsonl(out_dir / "belief_pairs.jsonl", belief_pairs)

    # 5-6. Type-1
    step_pairs = []
    if not args.skip_type1:
        print(f"[4/6] SFT proxy 샘플링 ({SFT_PROXY_MODEL} × {len(personas) * args.n_samples})")
        prediction_rows = sft_proxy_sampling(
            client, personas, args.question, args.gt_answer,
            args.problem_id, args.n_samples,
        )
        write_jsonl(out_dir / "predictions.jsonl", prediction_rows)

        print(f"[5/6] Type-1: locate + rectify + judge")
        step_pairs = build_type1_step_pairs(
            client, prediction_rows, persona_by_id, args.question, args.gt_answer,
        )
        write_jsonl(out_dir / "step_pairs.jsonl", step_pairs)

    # 7. merge
    print("[6/6] train.jsonl (merge)")
    write_jsonl(out_dir / "train.jsonl", step_pairs + belief_pairs)

    print()
    print("[done] 생성 완료.")
    print(f"       Type-1 step_pairs: {len(step_pairs)}")
    print(f"       Type-2 belief_pairs: {len(belief_pairs)}")
    print(f"       train.jsonl total: {len(step_pairs) + len(belief_pairs)}")


if __name__ == "__main__":
    main()
