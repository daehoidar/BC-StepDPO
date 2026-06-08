"""data_pipeline_fullstepdpo/4_train_fullstepdpo.py

Full-Step DPO Stage 4: per-step reward weighted DPO 학습.

3c 산출 chains_fullstepdpo.jsonl 입력:
  1) (problem_id × persona_id) 단위로 체인 그룹핑
  2) final_correct 기준으로 win / lose 체인 페어 구성
     (둘 다 맞거나 둘 다 틀린 경우 → avg r_math 기준 high vs low)
  3) 각 step t 마다 w_t = alpha * r_math_t + beta_w * r_persona_t
  4) L = -Σ_t w_t · log σ(β · Δ_t) 로 학습

Usage:
    accelerate launch data_pipeline_fullstepdpo/4_train_fullstepdpo.py \\
        --base-model checkpoints/sft_ref \\
        --chains data_pipeline_fullstepdpo/output/chains_fullstepdpo.jsonl \\
        --output checkpoints/fullstepdpo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from losses.bc_fullstepdpo_loss import bc_fullstepdpo_loss  # noqa: E402


@dataclass
class Example:
    win_input_ids: torch.Tensor
    win_attention_mask: torch.Tensor
    win_step_mask: torch.Tensor
    lose_input_ids: torch.Tensor
    lose_attention_mask: torch.Tensor
    lose_step_mask: torch.Tensor
    step_weight: float


def build_pairs(chains: list[dict]) -> list[tuple[dict, dict, int]]:
    """(problem_id × persona_id) 그룹 내에서 (win, lose, step_t) 트리플 생성.

    - correct vs incorrect 체인 우선 페어
    - 모두 같은 경우 avg r_math 기준 high vs low 폴백
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in chains:
        key = (c["problem_id"], c.get("persona_id", ""))
        groups[key].append(c)

    pairs = []
    for grp in groups.values():
        correct = [c for c in grp if c.get("final_correct")]
        incorrect = [c for c in grp if not c.get("final_correct")]

        if correct and incorrect:
            candidates = list(product(correct[:2], incorrect[:2]))
        else:
            sorted_grp = sorted(
                grp,
                key=lambda c: sum(s["r_math"] for s in c["chain"]) / max(1, len(c["chain"])),
                reverse=True,
            )
            candidates = [(sorted_grp[0], sorted_grp[-1])] if len(sorted_grp) >= 2 else []

        for w, l in candidates:
            min_len = min(len(w["chain"]), len(l["chain"]))
            for t in range(min_len):
                pairs.append((w, l, t))

    return pairs


def tokenize_step(
    tokenizer,
    persona_tag: str,
    problem: str,
    prefix_steps: list[str],
    step_text: str,
    max_len: int,
) -> dict:
    """prefix까지 step_mask=0, step 토큰만 step_mask=1."""
    prompt = (
        (f"{persona_tag}\n" if persona_tag else "")
        + f"Problem: {problem}\nSolution:\n"
        + ("\n".join(prefix_steps) + "\n" if prefix_steps else "")
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    step_ids = tokenizer(step_text, add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + step_ids)[:max_len]
    L = len(full_ids)
    step_start = min(len(prompt_ids), L)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * L,
        "step_mask": [0] * step_start + [1] * (L - step_start),
    }


class FullStepDPODataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[dict, dict, int]],
        tokenizer,
        max_len: int,
        alpha: float,
        beta_w: float,
    ):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len
        self.alpha = alpha
        self.beta_w = beta_w

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Example:
        win_chain, lose_chain, t = self.pairs[idx]
        persona_tag = win_chain.get("persona_tag", "")
        problem = win_chain["problem"]

        win_entry = win_chain["chain"][t]
        lose_entry = lose_chain["chain"][t]
        win_prefix = [s["step"] for s in win_chain["chain"][:t]]
        lose_prefix = [s["step"] for s in lose_chain["chain"][:t]]

        wt = tokenize_step(self.tok, persona_tag, problem,
                           win_prefix, win_entry["step"], self.max_len)
        lt = tokenize_step(self.tok, persona_tag, problem,
                           lose_prefix, lose_entry["step"], self.max_len)

        w = (self.alpha * win_entry.get("r_math", 1.0)
             + self.beta_w * win_entry.get("r_persona", 1.0))

        return Example(
            win_input_ids=torch.tensor(wt["input_ids"], dtype=torch.long),
            win_attention_mask=torch.tensor(wt["attention_mask"], dtype=torch.long),
            win_step_mask=torch.tensor(wt["step_mask"], dtype=torch.long),
            lose_input_ids=torch.tensor(lt["input_ids"], dtype=torch.long),
            lose_attention_mask=torch.tensor(lt["attention_mask"], dtype=torch.long),
            lose_step_mask=torch.tensor(lt["step_mask"], dtype=torch.long),
            step_weight=w,
        )


def collate(batch: list[Example], pad_id: int) -> dict:
    def pad(seqs: list[torch.Tensor], val: int) -> torch.Tensor:
        L = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), L), val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    return {
        "win_input_ids":       pad([b.win_input_ids for b in batch], pad_id),
        "win_attention_mask":  pad([b.win_attention_mask for b in batch], 0),
        "win_step_mask":       pad([b.win_step_mask for b in batch], 0),
        "lose_input_ids":      pad([b.lose_input_ids for b in batch], pad_id),
        "lose_attention_mask": pad([b.lose_attention_mask for b in batch], 0),
        "lose_step_mask":      pad([b.lose_step_mask for b in batch], 0),
        "step_weight": torch.tensor([b.step_weight for b in batch], dtype=torch.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--chains", required=True,
                    help="3c 산출 chains_fullstepdpo.jsonl")
    ap.add_argument("--config", default="configs/step_dpo.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL 정규화 강도")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="w_t 내 r_math 가중치")
    ap.add_argument("--beta-w", type=float, default=1.0,
                    help="w_t 내 r_persona 가중치")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("grad_accum", 4)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chains = []
    with open(args.chains, encoding="utf-8") as f:
        for line in f:
            chains.append(json.loads(line))
    print(f"[load] {len(chains)} chains")

    pairs = build_pairs(chains)
    print(f"[pair] {len(pairs)} (win, lose, step_t) examples")

    ds = FullStepDPODataset(
        pairs, tokenizer,
        max_len=args.max_len,
        alpha=args.alpha,
        beta_w=args.beta_w,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    )
    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            target_modules=cfg.get("lora_targets",
                                   ["q_proj", "v_proj", "o_proj", "k_proj"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)

    ref = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    grad_accum = cfg.get("grad_accum", 4)
    total_steps = args.epochs * len(loader) // grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.get("warmup_steps", 100),
        num_training_steps=total_steps,
    )

    policy, ref, optimizer, loader, scheduler = accelerator.prepare(
        policy, ref, optimizer, loader, scheduler
    )

    beta = cfg.get("beta", args.beta)
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(policy):
                out = bc_fullstepdpo_loss(policy, ref, batch, beta=beta)
                accelerator.backward(out["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad],
                        cfg.get("max_grad_norm", 1.0),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % 20 == 0:
                print(
                    f"[ep{epoch} step{global_step}/{total_steps}] "
                    f"loss={out['loss'].item():.4f} "
                    f"acc={out['accuracy'].item():.3f} "
                    f"delta={out['delta_mean'].item():.4f}"
                )
            global_step += 1

    if accelerator.is_main_process:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
