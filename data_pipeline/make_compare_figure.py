"""make_compare_figure.py — Figure 3: SFT vs +BC-StepDPO 정성 비교 패널.

실제 생성 결과(tests/sft_vs_trained.py 출력)에서 대표 2케이스를 골라
2열(SFT | +BC-StepDPO) 패널로 렌더. 특허/논문 도면용.
  (A) high_high: SFT는 8/2=40 오류로 581(틀림) → +BC는 8×2=16으로 413(정답)  [수학 정합성]
  (B) elem_low vs high_high: 같은 step을 저학년=평이한 말 / 고학년=수식  [수준별 화법]
"""
from __future__ import annotations
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# '$'를 수식(mathtext) 기호로 해석하지 않고 리터럴로 출력 (금액 $ 표기 보존).
plt.rcParams["text.parse_math"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAP = 44  # 한 줄 최대 글자수 (열 안에 들어오도록 강제)

CASE_A = {
    "title": "(A) Math consistency  —  high-school persona (high_high)",
    "sft": ["Step 1: trendy: 8 / 2 = 40 dollars",
            "Step 2: 25 + 18 + 40 = 83",
            "Step 3: 83 x 7 = 581",
            "Final answer: 581   [WRONG]"],
    "bc":  ["Step 1: trendy: 8 x 2 = 16 dollars",
            "Step 2: 25 + 18 + 16 = 59",
            "Step 3: 59 x 7 = 413",
            "Final answer: 413   [CORRECT]"],
}
CASE_B = {
    "title": "(B) Level-appropriate expression  —  same computation",
    "sft": ['[elem_low]  "... 5 x $5 = $25 ..."',
            '[high_high] "... (5 x 5 = 25) ..."'],
    "bc":  ['[elem_low]  "$6 multiplied by 3 equals $18"',
            '            -> plain, concrete wording',
            '[high_high] "(5 x 5 = 25), (8 x 2 = 16)"',
            '            -> symbolic / LaTeX'],
}

BLUE = "#1f4e79"; GREEN = "#2e7d32"
BOX = dict(boxstyle="round,pad=0.5", fc="#f4f6f9", ec="#c9d3e0", lw=1)


def _wrap(lines):
    out = []
    for ln in lines:
        w = textwrap.wrap(ln, WRAP, subsequent_indent="    ") or [""]
        out.extend(w)
    return "\n".join(out)


def _col(ax, x, header, lines, color):
    ax.text(x, 0.86, header, fontsize=10.5, fontweight="bold", color=color,
            ha="left", va="top", transform=ax.transAxes, family="monospace")
    ax.text(x, 0.66, _wrap(lines), fontsize=9, ha="left", va="top",
            transform=ax.transAxes, family="monospace", bbox=BOX)


def main():
    out = REPO_ROOT / "docs" / "figures" / "fig_sft_vs_bcstepdpo"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0))
    for ax, case in zip(axes, [CASE_A, CASE_B]):
        ax.axis("off")
        ax.set_title(case["title"], fontsize=12, loc="left", pad=10)
        _col(ax, 0.02, "SFT (Qwen3-1.7B)", case["sft"], BLUE)
        _col(ax, 0.53, "+ BC-StepDPO (ours)", case["bc"], GREEN)
    fig.suptitle("SFT  vs  + BC-StepDPO : math correction & level-appropriate expression",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}.png / .pdf")


if __name__ == "__main__":
    main()
