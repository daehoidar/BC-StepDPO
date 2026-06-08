#!/usr/bin/env bash
# Step DPO 전체 파이프라인 (Stage 0 ~ 5).
# 실행 위치: repo root

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY가 설정되어 있어야 합니다}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
OUT_DIR="data_pipeline/output"
CKPT_DIR="checkpoints"
N_PROBLEMS="${N_PROBLEMS:-1500}"
SOLS_PER_ROW="${SOLS_PER_ROW:-5}"
K_SAMPLES="${K_SAMPLES:-8}"

mkdir -p "$OUT_DIR/stepdpo" "$CKPT_DIR"

echo "=== Stage 0: Seed problem sampling ==="
python data_pipeline/0_seed_problems.py \
    --n-problems "$N_PROBLEMS" \
    --out "$OUT_DIR/seed_problems.jsonl"

echo "=== Stage 1: GPT-4o SFT 데이터 합성 ==="
python data_pipeline/1_synthesize_sft.py \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --solutions-per-row "$SOLS_PER_ROW" \
    --output "$OUT_DIR/sft_data.jsonl"

echo "=== Stage 2: Reference SFT ==="
accelerate launch data_pipeline/2_train_sft.py \
    --base-model "$BASE_MODEL" \
    --data "$OUT_DIR/sft_data.jsonl" \
    --output "$CKPT_DIR/sft_ref" \
    --config configs/default.yaml

echo "=== Stage 3 (Shared Sampling): π_ref 샘플링 + 페르소나 cascade ==="
python data_pipeline/shared_sampling.py \
    --ref-model "$CKPT_DIR/sft_ref" \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --k-samples "$K_SAMPLES" \
    --output "$OUT_DIR/samples_with_persona_labels.jsonl"

echo "=== Stage 3a: 최초 오류 스텝 검출 ==="
python data_pipeline_stepdpo/3_locate_first_error.py \
    --samples-path "$OUT_DIR/samples_with_persona_labels.jsonl" \
    --output "$OUT_DIR/stepdpo/located_errors.jsonl"

echo "=== Stage 3b: win/lose 페어 구성 ==="
python data_pipeline_stepdpo/4_build_pairs.py \
    --located "$OUT_DIR/stepdpo/located_errors.jsonl" \
    --output "$OUT_DIR/stepdpo/pairs_stepdpo.jsonl"

echo "=== Stage 4: BC-StepDPO 학습 ==="
accelerate launch data_pipeline/4_train_bc_stepdpo.py \
    --base-model "$CKPT_DIR/sft_ref" \
    --pairs "$OUT_DIR/stepdpo/pairs_stepdpo.jsonl" \
    --config configs/step_dpo.yaml \
    --output "$CKPT_DIR/bc_stepdpo"

echo "=== Stage 5: 평가 ==="
python data_pipeline/5_evaluate.py \
    --model "$CKPT_DIR/bc_stepdpo" \
    --test-set "$OUT_DIR/test.jsonl" \
    --personas-path personas.json \
    --output "$CKPT_DIR/bc_stepdpo/eval_results.json"

echo "Done. 결과: $CKPT_DIR/bc_stepdpo/eval_results.json"
