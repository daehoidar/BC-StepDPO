"""make_heldout_figures.py — held-out 결과 공유용 figure (팀 공유).
A: 모델별 Persona Cons + Belief-Flip 막대 (핵심 지표)
B: 기존(누수) vs held-out Final Acc (누수 영향)
출력 docs/figures_final/ (png+pdf, 영어 라벨).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ED = REPO / "eval_ho"
ORD = ["sft", "vanilla_dpo", "step_dpo", "type1_only", "full", "full_aug"]
LBL = {"sft": "SFT", "vanilla_dpo": "Vanilla\nDPO", "step_dpo": "Step-DPO",
       "type1_only": "BC (T1)", "full": "BC (T1+T2)", "full_aug": "BC (T2 aug)"}
PROPOSED = {"type1_only", "full", "full_aug"}


def M(name):  # metrics dict (0~1)
    return json.load(open(ED / f"{name}.json"))["metrics"]


def flip(name):  # belief_flip_accuracy (0~100)
    return json.load(open(ED / f"{name}_flip.json")).get("belief_flip_accuracy")


def final_acc(name):
    return 100 * M(name)["final_answer_accuracy"]


def persona_cons(name):  # = 1 - persona 오류율
    return 100 * (1 - M(name)["step_persona_err_rate"])


def _save(fig, base):
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {base}.png/.pdf")


def figA():
    pc = [persona_cons(n) for n in ORD]
    bf = [flip(n) for n in ORD]
    x = range(len(ORD)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    c1 = ["#1f4e79" if n in PROPOSED else "#9bb7d4" for n in ORD]
    c2 = ["#c55a11" if n in PROPOSED else "#e8b08a" for n in ORD]
    b1 = ax.bar([i - w/2 for i in x], pc, w, label="Persona Cons.", color=c1)
    b2 = ax.bar([i + w/2 for i in x], bf, w, label="Belief-Flip", color=c2)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6,
                f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([LBL[n] for n in ORD])
    ax.set_ylabel("score  [%]"); ax.set_ylim(0, 95)
    ax.set_title("Held-out (gpt-4o judge, n=60): Persona Cons. & Belief-Flip\n"
                 "(dark bars = proposed BC-StepDPO variants)")
    ax.legend(loc="upper right")
    _save(fig, str(REPO / "docs/figures_final/fig_heldout_metrics"))


def figB():
    models = ["SFT", "Vanilla", "Step-DPO", "BC(T1)", "BC(full)"]
    old_final = [90.3, 84.2, 82.5, 85.8, 82.2]  # 기존(누수, n=15)
    new_final = [final_acc(n) for n in ["sft", "vanilla_dpo", "step_dpo", "type1_only", "full"]]
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.bar([i - w/2 for i in x], old_final, w, label="old (leaked test, n=15)", color="#bbbbbb")
    ax.bar([i + w/2 for i in x], new_final, w, label="held-out (gpt-4o, n=60)", color="#1f4e79")
    for i, (o, nv) in enumerate(zip(old_final, new_final)):
        ax.text(i - w/2, o + 0.6, f"{o:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, nv + 0.6, f"{nv:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylabel("Final Acc.  [%]"); ax.set_ylim(0, 100)
    ax.set_title("Data-leakage inflation: Final Acc. drops ~15 pts on a true held-out set")
    ax.legend()
    _save(fig, str(REPO / "docs/figures_final/fig_leakage_impact"))


if __name__ == "__main__":
    (REPO / "docs/figures_final").mkdir(parents=True, exist_ok=True)
    figA(); figB()
