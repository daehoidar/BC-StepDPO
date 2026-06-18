"""make_result_figures.py — 특허/논문용 결과 figure 생성 (실측 데이터 기반).

Figure 1: BC-StepDPO 학습 곡선 (loss + accuracy vs step)  ← 학습 로그에서
Figure 2: 학습자 수준별 화법 reject 분포 (사유별 stacked bar) ← samples에서

출력: docs/figures/ 에 png + pdf (300 dpi). 라벨은 영어(폰트 안전).
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONA_ORDER = ["elem_low", "elem_high", "mid_low", "mid_high", "high_low", "high_high"]


def parse_train_log(path: Path):
    steps, loss, acc = [], [], []
    pat = re.compile(r"step(\d+)\] loss=([0-9.]+) acc=([0-9.]+)")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1))); loss.append(float(m.group(2))); acc.append(float(m.group(3)))
    return steps, loss, acc


def fig_training_curve(train_log: Path, out: Path, dpi: int):
    steps, loss, acc = parse_train_log(train_log)
    if not steps:
        print(f"[fig1] 학습 로그 파싱 실패: {train_log}"); return
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(steps, loss, color="#1f4e79", lw=1.8, label="Step-DPO loss")
    ax1.set_xlabel("Training step"); ax1.set_ylabel("Loss", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(steps, acc, color="#c00000", lw=1.4, alpha=0.8, label="Preference accuracy")
    ax2.set_ylabel("Preference accuracy", color="#c00000"); ax2.set_ylim(-0.05, 1.08)
    ax2.tick_params(axis="y", labelcolor="#c00000")
    ax1.set_title("BC-StepDPO training: loss ↓, preference accuracy ↑")
    fig.tight_layout()
    _save(fig, out, dpi)


def categorize(why: str) -> str:
    w = (why or "").lower()
    if "429" in why or "stage-c error" in w:
        return "API-error"
    if "variable" in w or "equation" in w or "'x'" in w or "algebra" in w:
        return "concept: variable/equation"
    if "percent" in w or "ratio" in w:
        return "concept: percentage/ratio"
    if "concrete" in w or "bare symbolic" in w or "화법" in why or "context" in w or "abstract" in w:
        return "expression-style: no concrete grounding"
    return "other"


CAT_ORDER = ["concept: variable/equation", "concept: percentage/ratio",
             "expression-style: no concrete grounding", "other"]
CAT_COLOR = {"concept: variable/equation": "#1f4e79",
             "concept: percentage/ratio": "#2e75b6",
             "expression-style: no concrete grounding": "#c55a11",
             "other": "#bfbfbf"}


def fig_reject_distribution(samples: Path, out: Path, dpi: int):
    tot = Counter()
    cat_by_persona = {p: Counter() for p in PERSONA_ORDER}
    for line in open(samples, encoding="utf-8"):
        r = json.loads(line)
        pid = r.get("persona_id")
        if pid not in cat_by_persona:
            continue
        for lab in r.get("step_persona_labels", []):
            tot[pid] += 1
            if lab.get("verdict") == "reject_persona":
                cat_by_persona[pid][categorize(lab.get("reasoning", ""))] += 1

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = range(len(PERSONA_ORDER))
    bottoms = [0.0] * len(PERSONA_ORDER)
    for cat in CAT_ORDER:
        vals = [100 * cat_by_persona[p].get(cat, 0) / max(tot[p], 1) for p in PERSONA_ORDER]
        ax.bar(x, vals, bottom=bottoms, color=CAT_COLOR[cat], label=cat, width=0.62)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels(PERSONA_ORDER, rotation=20)
    ax.set_ylabel("Persona-mismatch (reject) rate  [%]")
    ax.set_title("Level-appropriate expression: reject rate by learner level", pad=12)
    ax.set_ylim(0, max(bottoms) * 1.30)   # 막대 위 %라벨·범례가 상단과 겹치지 않게 여백 확보
    for i, b in enumerate(bottoms):
        ax.text(i, b + max(bottoms) * 0.015, f"{b:.0f}%", ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    _save(fig, out, dpi)


def _save(fig, out_base: Path, dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_base}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--samples", default="data_pipeline/output/samples_with_persona_labels.jsonl")
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    od = Path(args.out_dir)
    fig_training_curve(Path(args.train_log), od / "fig_training_curve", args.dpi)
    fig_reject_distribution(Path(args.samples), od / "fig_reject_distribution", args.dpi)


if __name__ == "__main__":
    main()
