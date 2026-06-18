# Full-Step-DPO를 eval matrix에 추가 — 실행 계획 (실행 보류 중)

작성 2026-06-17. 사용자 지시로 **계획·스크립트만 준비, 실행 보류**. 현재 5모델 결과표는 유지.

## 1. 결론 (feasibility): 추가 **가능** ✅
- 팀원 `data_pipeline_fullstepdpo/`(GitHub daehoidar/BC-StepDPO main, `run_pipeline.sh` 신규 + `4_train_fullstepdpo.py` 업데이트)를 프로젝트로 복사 완료, py_compile·임포트 검증 통과.
- `4_train_fullstepdpo.py` = 같은 base 위 **LoRA adapter** 산출 → merge → 기존 `5_evaluate.py` + `eval_belief_flip.py`로 **동일 test set 평가** 가능 → 6번째 행.
- 우리 `utils`/`persona_verifier`(화법 프롬프트)를 그대로 사용. 3c→4 스키마(`r_math`/`r_persona`) 일치.

## 2. 단, "step-dpo처럼 단순 재학습"은 아님
Full-Step-DPO는 정의상 **자체 PRM 데이터 파이프라인**이 선행 (chosen/rejected 페어가 아니라 per-step reward):
```
3a MC rollout (각 step prefix에서 M회 롤아웃 → step_value)   ← GPU 생성, 최대 비용
3b PRM 학습 (2-head: r_math + r_persona)
3c PRM 스코어 + chain 패킹 → chains_fullstepdpo.jsonl
4  per-step weighted DPO (chain win/lose, α·r_math + β·r_persona 가중) → LoRA
merge → 5_evaluate + eval_belief_flip → aggregate
```

## 3. 준비된 산출물
- 스크립트: **`scripts/fullstepdpo_campaign_slurm.sh`** (제출 안 함). 모든 함정 반영:
  - base = `checkpoints/sft_qwen3_1.7b_eos_merged` (공정 비교, 팀원 0.6B 기본값 대신 1.7B)
  - 개인 키, `VLLM_USE_FLASHINFER_SAMPLER=0`, `CUDA_HOME`, judge=gpt-4o-mini, `--disable-stage-b`
  - continue-on-error + skip-if-exists (watchdog 재제출 시 이어감)
  - 스케일 env: `M_ROLLOUTS`(기본8), `K_SAMPLES`(기본8), `N_SUB`(3a samples 행 제한, 0=전체)
- 기존 fullstepdpo 백업: `data_pipeline_fullstepdpo_old_backup/`

## 4. 실행할 때 (승인 후)
1. **(권장) 먼저 소규모 스모크**로 팀원 3a~3c가 우리 1.7B에서 도는지 확인:
   ```
   N_SUB=8 M_ROLLOUTS=2 K_SAMPLES=4 sbatch scripts/fullstepdpo_campaign_slurm.sh
   ```
   3a→3b→3c→4가 에러 없이 chains/adapter를 만들면 OK. 버그 있으면 로그로 수정.
2. **본 실행** (공정 스케일):
   ```
   M_ROLLOUTS=8 K_SAMPLES=8 sbatch scripts/fullstepdpo_campaign_slurm.sh
   ```
   watchdog 붙이려면 `scripts/watchdog_*` 패턴 참고(잡 이름 `fsdpo-camp`).
3. **aggregate에 6번째 행 추가** (`data_pipeline/aggregate_results.py` `MODELS` 리스트):
   ```python
   MODELS = [
       ("sft",         "SFT (Baseline)"),
       ("vanilla_dpo", "Vanilla DPO"),
       ("step_dpo",    "Step-DPO"),
       ("type1_only",  "BC-StepDPO (Type-1 only)"),
       ("full",        "Full BC-StepDPO (Type-1 + Type-2)"),
       ("fullstepdpo", "Full-Step-DPO (PRM)"),   # ← 추가
   ]
   ```
   (campaign 스크립트 끝에서 aggregate 호출하므로, 이 한 줄만 추가하면 6행 표 생성. **현재 5모델 표 보존을 위해 지금은 추가하지 않음** — 실행 결정 시 추가.)

## 5. 리스크 / 결정 필요
- **MC rollout(3a) 비용**: M·K·step수에 비례. 전체 samples면 수 시간. `N_SUB`/`M_ROLLOUTS` 축소로 조절.
- 팀원 3a~3c는 우리 1.7B + merged SFT 경로에서 **실제 실행 미검증** → 스모크 먼저.
- 3a/3c verifier가 GPT(StageC) 호출 → 개인키·gpt-4o-mini로 통일(스크립트 반영). stage-b(로컬 vLLM 서버)는 비활성.
- **명명**: 표에서 "Full BC-StepDPO"(우리)와 "Full-Step-DPO (PRM)"(팀원)는 다른 방법 — 혼동 주의.
- PRM base 모델 선택(현재 SFT_MERGED). 필요시 raw Qwen3-1.7B로 변경 검토.

## 6. 예상 시간
스모크 ~20분, 본 실행 ~3~5시간(MC rollout 병목). 오늘 내 완주 가능.
