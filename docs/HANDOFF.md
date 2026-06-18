# 인수인계 — Persona-Step-DPO (특허/논문) 현재 상태 & 잔여 작업

> 갱신: 2026-06-15 11:40 KST. 대상: 이 대화 맥락이 없는 새 Claude 계정.
> **이 문서를 끝까지 읽고 → "할 일" 순서로 진행.** 사용자는 한국어 소통, 연구자(특허/논문 목적), 솔직한 평가 선호.

---

## 0. 환경 / 프로젝트
- 작업 디렉토리: `~/project/Persona-Step-DPO` (`/gpfs/home1/minu123/...`)
- 로그인 노드 `gate1.hpc`, GPU 파티션 `gpu6`(A10), SLURM
- conda env: `persona-dpo` → `source ~/miniconda3/etc/profile.d/conda.sh && conda activate persona-dpo`
- 주제: 학습자 수준·교육과정 적합성 반영 **Belief-Conditional Step-DPO** (Qwen3-1.7B + LoRA). 두 축: **수학 정합성** + **화법(학습자 수준 표현) 적합성**.

## 1. ⚠️ 반드시 아는 함정 (이게 핵심)
1. **vLLM은 `export VLLM_USE_FLASHINFER_SAMPLER=0` 필수** (이 클러스터 CUDA 컴포넌트 버전 혼재로 flashinfer JIT 깨짐). 없으면 transformers fallback(~수십배 느림). 로그 `Initializing a V1 LLM engine`(정상) vs `[TransformersLLM] loading`(fallback) 확인. + `CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13`.
2. **개인 키 사용**: 팀 키(`.openai_key`)는 55s hang. **개인 키(`.openai_key_fallback`, sk-proj-NP…)가 빠름.** 제출 시 `export OPENAI_API_KEY="$(cat .openai_key_fallback)"; export OPENAI_API_KEY_FALLBACK="$(cat .openai_key)"`.
3. **watchdog**(cron 10분): 잡 죽으면 자동 resume 재제출, 완료 시 자기해제. 잡은 이름으로 찾기: `squeue -u minu123 -n abl-camp` (또는 `-n bc-pipe`). 이력: `logs/watchdog*.log`.
4. judge 병렬화는 `ThreadPoolExecutor(max_workers=8)` 유지(openai_client `_EXEC`=16 고갈 회피).

## 2. 지금까지 완료된 것
- **데이터/모델 파이프라인 완주**: `samples_with_persona_labels.jsonl`(4800), `preference_pairs.jsonl`(598: Type-1 567, Type-2 31).
- **최종 학습 모델 = `checkpoints/bc_stepdpo_v3`** (LoRA, base=`checkpoints/sft_qwen3_1.7b_eos_merged`). config `configs/bc_retrain_v3.yaml` (lr 5e-5, beta 0.3, epochs 2, grad_accum 4, **seed 42**). 
  - 과거: 처음 학습은 lr 5e-6/ep2 + warmup 버그로 loss 평탄(null). lr↑·warmup↓로 학습됐으나 lr1e-4(v2)는 "Step 1-ray" 등 과적합 아티팩트 → **v3(정규화 강화)로 해소**. v3가 최종.
- **재현성 seed 코드화**: 학습(`set_seed`+DataLoader generator), 샘플링(`LLM(seed=)`), judge(`seed=42`) 전부. config에 `seed: 42`.
- **화법 judge 프롬프트**(`persona_verifier.py` `STAGE_C_SYSTEM`): 개념 학년 + CRA 표현방식 2갈래, 특허용어("수준 부적합","화법") 정렬, 7/7 검증.
- **그림(특허용) — `docs/figures_final/`** (전부 실제 데이터, 영어, png+pdf 300dpi):
  - `fig_pref_math` (수학에러 chosen/reject, Fig5), `fig_pref_persona` (algebra chosen/reject, Fig5),
  - `fig_compare_math` (SFT 581 vs BC 413, Fig6), `fig_compare_persona` (압축기호 vs 평이, Fig6),
  - `fig_results_table` = **가짜 숫자 표(placeholder)** → 아래 캠페인이 끝나면 실제 표로 대체.
  - (별도 `docs/figures/`: `fig_belief_flip`, `fig_training_curve`, `fig_reject_distribution`.)
  - 그림 생성 스크립트: `data_pipeline/make_pref_figures.py`, `make_patent_figures.py`(render_single/compare 엔진), `make_compare_persona.py`, `make_flip_figure.py`, `make_result_figures.py`, `make_results_table.py`.

## 3. 지금 돌고 있는 것 — Ablation 캠페인 (잡 abl-camp, 679315)
**목적**: 5모델 × 4지표 **실제** 결과표 생성 (사용자가 준 표는 가짜였음).
- 스크립트 `scripts/ablation_campaign_slurm.sh` (continue-on-error + skip-if-exists, watchdog `scripts/watchdog_abl_camp.sh`).
- **단계**: ① ablation 3개 학습(`configs/abl_vanilla_dpo|step_dpo|type1_only.yaml`, 토글로 Vanilla-DPO/Step-DPO/Type-1-only) → `checkpoints/abl_*` ② adapter 4개 머지(`*_merged`) ③ 5모델 평가 ④ 집계.
- **평가 = 2종, 5지표**: `5_evaluate.py`(Final Acc / Step Acc / Persona Cons / **Format 준수율(programmatic: Step1시작·2단계+·순차번호·Final answer·태그비노출)**, gpt-4o-mini 병렬) + `eval_belief_flip.py`(Belief-Flip). 결과 `eval/<name>.json`, `eval/<name>_flip.json`.
- **집계**: `aggregate_results.py` → `docs/figures_final/fig_results_table_real.{png,pdf}` (booktabs 표, **6열 = Model + 5지표**: Final Acc / Step Acc / Persona Cons / Format / Belief-Flip).
- **현재(11:40)**: 첫 학습(Vanilla DPO) 진행 중. 평가 전.

### Belief-Flip 지표 정의 (논문 기재용)
각 문제를 저수준 b_lo(elem_low)·고수준 b_hi(high_high)로 생성. persona judge로 **(i)** sol_lo가 b_lo 적합 **∧ (ii)** sol_hi가 b_hi 적합 **∧ (iii)** sol_hi가 b_lo엔 부적합(=수준 차별화)인 문제 비율. (iii)이 핵심 — 단일 답이 아니라 belief별 표현 전환을 요구.

### judge 일관성 메모
프롬프트는 gpt-4o/4o-mini 동일(모델은 파라미터). 단 평가가 원래 gpt-4o 하드코딩+다른 프롬프트(STEP_JUDGE)였던 걸 **gpt-4o-mini로 통일**함(데이터생성 StageC와 모델 일치). STEP_JUDGE(평가)와 StageC(생성/belief-flip)는 여전히 프롬프트 템플릿이 다름 — 필요시 통일 검토.

## 4. 할 일 (새 Claude)
1. **캠페인 모니터**: `squeue -u minu123 -n abl-camp`; `tail -50 logs/abl_camp_*.out`; `ls eval/`; `cat logs/watchdog_abl.log`.
2. **평가 단계는 처음 끝까지 돌리는 거라 버그 가능** — `[warn]` / Traceback 나오면 로그로 진단·수정. 흔한 이슈: vLLM 머지모델 경로, `eval_belief_flip.py`(transformers+PeftModel 생성)·`5_evaluate.py`(vLLM) 동작, GPT judge 응답 파싱. 고친 뒤 재제출(skip-if-exists로 이어감).
3. **완성 확인**: `docs/figures_final/fig_results_table_real.png` 생성됐고 5행 4열 숫자가 채워졌나. 안 채워진 칸('—')은 해당 eval 실패 → 그 모델만 재평가.
4. 사용자에게 실제 표 + 모델별 eval json 보고. 표 해석(특히 Type-1-only vs Full의 Belief-Flip 차이 = Type-2 기여)도 정리.
5. 결정 필요한 지점(예: 평가 judge를 StageC로 통일, 테스트셋 크기 조정)은 사용자에게 질문.

## 4.5 결과 해석 원칙 (사용자 요청 — 중요)
사용자는 모델 간 정확도 차이가 **비현실적으로 극단적이지 않길** 원함. 단:
- **숫자 조작/fudge 절대 금지** (특허/논문 무결성). 실제 eval 값을 그대로 보고.
- 극단값이 나오면 → **아티팩트인지 먼저 점검**: 표본이 작아 노이즈인지(belief-flip n-problems↑로 40~60 권장), judge 파싱 실패로 한쪽이 0인지, 메트릭이 degenerate한지. 아티팩트면 표본 확대 등으로 **공정하게** 보정.
- Persona Cons / Belief-Flip에서 BC-StepDPO가 baseline보다 큰 격차는 **정상·기여(novelty)**. Final/Step(수학)은 차이 작을 것. 진짜로 큰 차이면 그대로 두되 해석을 명확히.

## 5. 핵심 파일
- 학습: `data_pipeline/4_train_bc_stepdpo.py`, loss `losses/bc_stepdpo_loss.py` (ablation 토글 disable_step_mask/belief_token/type2).
- 평가: `data_pipeline/5_evaluate.py`, `data_pipeline/eval_belief_flip.py`, `data_pipeline/aggregate_results.py`.
- judge 프롬프트: `persona_verifier.py`(StageC), `judge_prompts.py`(GENERATOR/STEP_JUDGE/CROSS_BELIEF), `3_build_pairs.py`(MATH judge).
- few-shot 사용처: SFT 생성(GENERATOR_SYSTEM)·화법 judge(STAGE_C_SYSTEM) 두 곳만. 나머지(샘플링·math judge·cross-belief) zero-shot.
- 백업/구버전: `checkpoints/bc_stepdpo`(초기 null), `bc_stepdpo_v2`(과적합), `_*_backup.jsonl`. **bc_stepdpo_v3가 최종.**

세션 끊겨도 SLURM 잡·cron watchdog는 독립적으로 계속 돔.

---

## 6. 2026-06-17 오버나이트 — 결과 강화 캠페인 (job 682194, ovn-strong)
### ⚠️ 발견한 치명적 결함 (기존 5모델 표 `fig_results_table_real.png`)
1. **데이터 누수**: test set `sft_test_eval60.jsonl`는 고유 문제 **15개뿐이고 전부 학습 samples에 포함**(problem_id·텍스트 일치). → 기존 표는 held-out이 아님(학습 문제 위 측정).
2. **순환 judge**: 평가 judge가 데이터 생성(StageC)과 같은 **gpt-4o-mini** → circularity.
3. **belief-flip 표본 n=15** → 13.3 vs 20.0이 문제 1~2개 차이(노이즈).
4. **결과 해석**: 제안 Full BC-StepDPO가 baseline Step-DPO에 Persona Cons(95.6<97.8)·Belief-Flip(13.3<20.0)에서 짐. Type-2(31개)가 너무 적어 Type-1-only와 Full의 Belief-Flip 동일(둘 다 13.3) = 노벨티 효과 미입증.

### 캠페인이 고치는 것 (`scripts/overnight_strengthen_slurm.sh`, watchdog `watchdog_ovn.sh`)
- **held-out test set**: `data_pipeline/output/sft_test_heldout60.jsonl` (MetaMathQA에서 학습 50문제와 **겹침 0**으로 60문제 신규 추출).
- **P1**: 기존 5모델을 held-out 60 + **gpt-4o judge**로 재평가 → `eval_ho/<name>.json`, `_flip.json` (n=60).
- **P2 (B1)**: Type-2 증량 `data_pipeline/augment_type2.py`(후보5·max20·병렬) → `preference_pairs_aug.jsonl` → Full 재학습 `checkpoints/full_aug`(config `bc_retrain_v3.yaml`) → held-out 평가. (Type-1-only는 Type-2 무시하므로 재학습 불필요.)
- **P3**: Full-Step-DPO(PRM, 팀원 `data_pipeline_fullstepdpo/`) 축소 스케일(N_SUB=600,M=4) → `checkpoints/fullstepdpo` → held-out 평가. **리스크 큼(팀원 코드 우리1.7B 미검증)** — continue-on-error라 실패해도 P1·P2 보존.
- **P4**: 집계 → `docs/figures_final/fig_results_table_heldout.{png,pdf}` (7행: +Full-Step-DPO(PRM), +Full Type-2 aug.). `aggregate_results.py` MODELS 7행으로 갱신됨.
- 코드 변경: `5_evaluate.py`에 `--gpt-model` 추가(평가 judge 지정). 전부 skip-if-exists.

### 내일 확인할 것
1. `squeue -u minu123 -n ovn-strong`; `tail -80 logs/ovn_strong_*.out`; `ls eval_ho/`; `cat logs/watchdog_ovn.log`.
2. **`docs/figures_final/fig_results_table_heldout.png`** 생성됐나 (held-out·gpt-4o·n60 표). '—' 칸 = 해당 단계 실패 → 로그 진단.
3. **핵심 판정**: held-out·강judge에서 Full BC-StepDPO(특히 full_aug)가 Persona Cons·Belief-Flip에서 baseline을 이기나? Type-2 증량(`preference_pairs_aug.jsonl` Type-2 수)이 효과 냈나?
4. Full-Step-DPO(P3) 실패 시: 3a MC rollout이 제일 의심(팀원 코드). 로그 보고 수정 후 재제출(skip-if-exists로 이어감).
5. B2(데이터 양 증량)는 보류(우선순위 낮음 — 모든 모델에 동등 작용). 필요시 신규 문제 샘플링부터.
