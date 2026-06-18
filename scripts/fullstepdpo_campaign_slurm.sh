#!/usr/bin/env bash
#SBATCH --job-name=fsdpo-camp
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/fsdpo_camp_%j.out
#SBATCH --error=logs/fsdpo_camp_%j.err

# Full-Step-DPO(PRM 기반)를 우리 ablation matrix에 6번째 행으로 추가하는 캠페인.
# ⚠️ 아직 제출하지 말 것 — 사용자 승인 후 sbatch. (계획 단계 산출물)
#
# 흐름: 3a MC rollout → 3b PRM 학습 → 3c PRM score/pack → 4 Full-Step-DPO 학습
#       → merge → 5_evaluate + eval_belief_flip → aggregate(6모델 표)
# 공정 비교: 모든 단계 base = 우리 1.7B SFT(merged), 기존 samples·seed·test 재사용.
# continue-on-error(|| echo warn) + skip-if-exists 로 watchdog 재제출 시 이어감.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
cd ~/project/Persona-Step-DPO

# ── 환경 (함정 전부 반영) ────────────────────────────────────────────────
export OPENAI_API_KEY="$(cat .openai_key_fallback)"     # 개인 키(빠름)
export OPENAI_API_KEY_FALLBACK="$(cat .openai_key)"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CU=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0                      # vLLM 필수

# ── 파라미터 (스케일 조절) ───────────────────────────────────────────────
SFT_MERGED=checkpoints/sft_qwen3_1.7b_eos_merged         # 공정 비교 base (1.7B)
TEST=data_pipeline/output/sft_test_eval60.jsonl
SEED=data_pipeline/output/seed_problems.jsonl
SAMPLES=data_pipeline/output/samples_with_persona_labels.jsonl
OUT=data_pipeline/output/fullstepdpo
M_ROLLOUTS="${M_ROLLOUTS:-8}"      # MC rollout 횟수 (비용 ↑). 축소판=4
K_SAMPLES="${K_SAMPLES:-8}"
N_SUB="${N_SUB:-0}"                # 3a용 samples 행 제한(0=전체). 축소판=일부
GPT_MODEL=gpt-4o-mini             # judge(개인키), 데이터생성 StageC와 동일
mkdir -p "$OUT" eval checkpoints

# 3a 입력 samples 서브셋(스케일 축소 시)
SAMP_IN="$SAMPLES"
if [ "$N_SUB" -gt 0 ]; then SAMP_IN="$OUT/_samples_sub.jsonl"; head -n "$N_SUB" "$SAMPLES" > "$SAMP_IN"; fi

# ── Stage 3a: MC rollout 라벨 (제일 비쌈) ────────────────────────────────
if [ ! -f "$OUT/mc_labeled.jsonl" ]; then
  echo "=== [3a] MC rollout (M=$M_ROLLOUTS) ==="
  python data_pipeline_fullstepdpo/3a_mc_rollout_label.py \
    --ref-model "$SFT_MERGED" --samples-path "$SAMP_IN" \
    --m-rollouts "$M_ROLLOUTS" --k-samples "$K_SAMPLES" \
    --disable-stage-b --gpt-model "$GPT_MODEL" \
    --output "$OUT/mc_labeled.jsonl" || echo "[warn] 3a 실패"
else echo "[skip] 3a (mc_labeled.jsonl 존재)"; fi

# ── Stage 3b: PRM 학습 ───────────────────────────────────────────────────
if [ ! -f checkpoints/prm/adapter_model.safetensors ] && [ ! -f checkpoints/prm/model.safetensors ]; then
  echo "=== [3b] PRM 학습 ==="
  accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_fullstepdpo/3b_train_prm.py \
    --base-model "$SFT_MERGED" --train-data "$OUT/mc_labeled.jsonl" \
    --output checkpoints/prm || echo "[warn] 3b 실패"
else echo "[skip] 3b (prm 존재)"; fi

# ── Stage 3c: PRM 스코어 + chain 패킹 ────────────────────────────────────
if [ ! -f "$OUT/chains_fullstepdpo.jsonl" ]; then
  echo "=== [3c] PRM score + pack ==="
  python data_pipeline_fullstepdpo/3c_score_and_pack.py \
    --ref-model "$SFT_MERGED" --prm-model checkpoints/prm --prm-base-model "$SFT_MERGED" \
    --seed-problems "$SEED" --k-samples "$K_SAMPLES" \
    --disable-stage-b --gpt-model "$GPT_MODEL" \
    --output "$OUT/chains_fullstepdpo.jsonl" || echo "[warn] 3c 실패"
else echo "[skip] 3c (chains 존재)"; fi

# ── Stage 4: Full-Step-DPO 학습 ──────────────────────────────────────────
if [ ! -f checkpoints/fullstepdpo/adapter_model.safetensors ]; then
  echo "=== [4] Full-Step-DPO 학습 ==="
  accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_fullstepdpo/4_train_fullstepdpo.py \
    --base-model "$SFT_MERGED" --chains "$OUT/chains_fullstepdpo.jsonl" \
    --config configs/step_dpo.yaml --output checkpoints/fullstepdpo || echo "[warn] 4 실패"
else echo "[skip] 4 (fullstepdpo adapter 존재)"; fi

# ── merge ────────────────────────────────────────────────────────────────
if [ ! -f checkpoints/fullstepdpo_merged/config.json ]; then
  echo "=== merge ==="
  python data_pipeline/merge_adapter.py --base-model "$SFT_MERGED" \
    --adapter checkpoints/fullstepdpo --output checkpoints/fullstepdpo_merged || echo "[warn] merge 실패"
else echo "[skip] merge"; fi

# ── 평가 (다른 모델과 동일) ──────────────────────────────────────────────
echo "=== EVAL fullstepdpo ==="
python data_pipeline/5_evaluate.py --model checkpoints/fullstepdpo_merged \
  --test-set "$TEST" --personas-path personas.json \
  --output eval/fullstepdpo.json || echo "[warn] eval 실패"
python data_pipeline/eval_belief_flip.py --merged "$SFT_MERGED" \
  --adapter checkpoints/fullstepdpo --test-set "$TEST" --n-problems 20 \
  --persona-low elem_low --persona-high high_high \
  --output eval/fullstepdpo_flip.json || echo "[warn] flip 실패"

# ── 집계 (6모델 표) ──────────────────────────────────────────────────────
# aggregate_results.py MODELS 에 ("fullstepdpo","Full-Step-DPO (PRM)") 추가 후 실행.
echo "=== aggregate (6모델) ==="
python data_pipeline/aggregate_results.py \
  --output docs/figures_final/fig_results_table_real.png || echo "[warn] aggregate 실패"
echo "=== fullstepdpo 캠페인 done ==="
