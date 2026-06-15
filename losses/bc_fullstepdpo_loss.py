"""losses/bc_fullstepdpo_loss.py

Full-Step DPO 손실 함수.

bc_stepdpo_loss.py와의 차이:
- per-step reward weight w_t 를 손실에 곱함
- 입력 batch는 dict 형태 (step_weight 필드 포함)

수식:
  L = -E_t[ w_t · log σ(β · Δ_t) ]

  Δ_t = [log π_θ(s_t^win) - log π_ref(s_t^win)]
       - [log π_θ(s_t^lose) - log π_ref(s_t^lose)]

  w_t = alpha * r_math_t + beta_w * r_persona_t
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def step_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    """step_mask가 1인 토큰들의 log p_model(token | prefix)의 합.

    Returns: (B,) log probability of the step tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    step_mask_shifted = step_mask[:, 1:].float()

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * step_mask_shifted).sum(dim=-1)


def bc_fullstepdpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Full-Step DPO 손실 계산.

    Args:
        policy_model: π_θ (LoRA 학습 중)
        ref_model:    π_ref (frozen)
        batch: dict with keys:
            win_input_ids, win_attention_mask, win_step_mask,
            lose_input_ids, lose_attention_mask, lose_step_mask,
            step_weight  (B,)  — w_t = alpha*r_math + beta_w*r_persona
        beta: KL 정규화 강도 (상수)

    Returns:
        dict with keys: loss, accuracy, delta_mean
    """
    win_lp_policy = step_logprob(
        policy_model, batch["win_input_ids"],
        batch["win_attention_mask"], batch["win_step_mask"],
    )
    lose_lp_policy = step_logprob(
        policy_model, batch["lose_input_ids"],
        batch["lose_attention_mask"], batch["lose_step_mask"],
    )

    with torch.no_grad():
        win_lp_ref = step_logprob(
            ref_model, batch["win_input_ids"],
            batch["win_attention_mask"], batch["win_step_mask"],
        )
        lose_lp_ref = step_logprob(
            ref_model, batch["lose_input_ids"],
            batch["lose_attention_mask"], batch["lose_step_mask"],
        )

    delta = (win_lp_policy - win_lp_ref) - (lose_lp_policy - lose_lp_ref)
    w = batch["step_weight"]
    per_sample_loss = -F.logsigmoid(beta * delta)
    loss = (w * per_sample_loss).mean()

    return {
        "loss": loss,
        "accuracy": (delta > 0).float().mean(),
        "delta_mean": delta.mean(),
    }
