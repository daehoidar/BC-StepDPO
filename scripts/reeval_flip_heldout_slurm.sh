#!/usr/bin/env bash
#SBATCH --job-name=reeval-flip
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/reeval_flip_%j.out
#SBATCH --error=logs/reeval_flip_%j.err

# (a) belief-flip 재평가: held-out 60문제(누수 없음) × n=60 으로 5모델.
# 결과 → eval/<name>_flip_heldout.json (원본 _flip.json 보존, 비교용).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
cd ~/project/Persona-Step-DPO
export OPENAI_API_KEY="$(cat .openai_key_fallback)"; export OPENAI_API_KEY_FALLBACK="$(cat .openai_key)"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

SFT=checkpoints/sft_qwen3_1.7b_eos_merged
TEST=data_pipeline/output/sft_test_heldout60.jsonl
N=60
mkdir -p eval

flip() {  # $1=name $2=adapter(or NONE)
  local aflag=""; [ "$2" != "NONE" ] && aflag="--adapter $2"
  echo "=== FLIP [$1] (n=$N, held-out) ==="
  python data_pipeline/eval_belief_flip.py --merged "$SFT" $aflag \
    --test-set "$TEST" --n-problems "$N" \
    --persona-low elem_low --persona-high high_high \
    --output "eval/${1}_flip_heldout.json" || echo "[warn] flip $1 실패"
}
flip sft         NONE
flip vanilla_dpo checkpoints/abl_vanilla_dpo
flip step_dpo    checkpoints/abl_step_dpo
flip type1_only  checkpoints/abl_type1_only
flip full        checkpoints/bc_stepdpo_v3
echo "=== reeval-flip done ==="
