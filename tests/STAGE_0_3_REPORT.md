# Stage 0~3 테스트 및 정리 보고서

본 문서는 데이터 파이프라인의 **Stage 0~3**(seed → SFT data → SFT 학습 → 선호 페어)을
*소규모*로 돌려서 *오류 없이 도는지* 확인하고, 각 결과를 한 곳에서 정리하기 위한
가이드 + 템플릿이다.

> 학습 본 단계인 Stage 4(BC-StepDPO 학습)와 Stage 5(평가)는 본 보고서 범위 밖.
> 별도 문서로 작성 예정.

---

## 0. 한눈에 보기

| Phase | 단계 | 명령 | 출력 | 비용·시간 (Mac M2 16GB 기준) |
|---|---|---|---|---|
| **A** | Stage 0 + 1 | `bash tests/run_sft_data.sh` | `tests/output/sft_data/` | GPT-4o ~24회 / ~$0.30 / 1-2분 |
| **B** | Stage 2 | `bash tests/run_sft_train.sh` | `tests/output/sft_train/` | 0회 / ~3-5분 |
| **C-1** | Stage 3 (Step-DPO) | `bash tests/run_pairs.sh step_dpo` | `tests/output/pairs_step_dpo/` | GPT-4o ~24회 / ~$0.50 / 3-5분 |
| **C-2** | Stage 3 (Full) | `bash tests/run_pairs.sh full` | `tests/output/pairs_full/` | GPT-4o ~24회 / ~$0.50 / 3-5분 |

**총합**: ~$1.3, ~15분.

---

## 1. 사전 준비

```bash
cd Persona-Step-DPO
pip install -r requirements.txt          # transformers, trl, peft, datasets, openai
export OPENAI_API_KEY=sk-...              # Stage 1·3 GPT-4o 호출용

# (선택) Mac MPS 가속 확인
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

vLLM이 없는 환경(Mac M-series)에선 `inference_backend.py`의 transformers fallback이
자동 적용 — 추가 작업 없음.

---

## 2. Phase A — SFT 데이터 만들기 (Stage 0 + 1)

### 무엇을 검증하나
- Stage 0: HuggingFace에서 MetaMathQA-40K 다운로드 + GSM_ 필터 + query dedupe가 정상 동작
- Stage 1: GPT-4o가 페르소나별로 풀이를 합성하고, 결과 jsonl이 schema대로 저장

### 실행

```bash
# 기본값 (2 문제 × 6 페르소나 × 2 풀이 = 24행)
bash tests/run_sft_data.sh

# 규모 조절
N_PROBLEMS=5 SOLS_PER_ROW=3 bash tests/run_sft_data.sh
```

### 산출 파일

| 파일 | 내용 |
|---|---|
| `tests/output/sft_data/seed_problems.jsonl` | 시드 문제 × 6 페르소나 복제 |
| `tests/output/sft_data/sft_data.jsonl` | GPT-4o 합성 풀이 (한 행 = 한 풀이) |
| `tests/output/sft_data/REPORT.md` | **자동 생성 결과 요약** |

### REPORT.md에 들어가는 것
- Stage 0: total rows, augmentation_type 분포, 페르소나별 분포, 샘플 행 1개
- Stage 1: 합성된 행 수, 페르소나별 분포, 평균 step 수, 평균 풀이 길이, 페르소나당 샘플 1개
- **Pass/Fail Verdict**: 두 파일 모두 만들어졌고 행 수가 기대치인가

### 점검 포인트
- [ ] sft_data.jsonl의 6 페르소나 모두 row가 있나? (한쪽이 비면 합성 실패)
- [ ] 페르소나별 풀이 톤이 *읽기에* 구별되나? (elem_low는 친근한 비유, high_high는 정형 표기 등)
- [ ] 풀이가 `\boxed{...}`로 끝나나? (출력 형식 준수 — 추후 정답 추출에 필요)
- [ ] 평균 step 수가 페르소나에 맞나? (하위권은 step 많음, 상위권은 적음)

---

## 3. Phase B — SFT 학습 sanity (Stage 2)

### 무엇을 검증하나
- Phase A 결과(sft_data.jsonl)로 Qwen3-0.6B + LoRA 1 epoch 학습이 *오류 없이 완주*
- 학습 loss가 NaN/Inf 없이 정상
- LoRA adapter 파일이 디스크에 저장

> 모델 품질 추구가 아니라 *파이프라인 sanity*만. 24행으로는 의미 있는 학습 X.

### 실행

```bash
bash tests/run_sft_train.sh

# Base 모델 변경
BASE_MODEL=Qwen/Qwen3-0.6B EPOCHS=1 bash tests/run_sft_train.sh
```

### 산출 파일

| 파일 | 내용 |
|---|---|
| `tests/output/sft_train/checkpoint/` | LoRA adapter (`adapter_model.safetensors` 등) |
| `tests/output/sft_train/training_log.txt` | 학습 stdout/stderr 전체 |
| `tests/output/sft_train/REPORT.md` | **자동 생성 결과 요약** |

### REPORT.md에 들어가는 것
- 학습 종료 exit code (0이면 OK)
- training_log.txt에서 loss 출력 횟수, first/last loss 값, NaN/Inf 발생 카운트
- 체크포인트 폴더의 파일 크기·adapter 파일 존재 여부
- log tail 마지막 10라인
- **Pass/Fail Verdict**: exit 0 + adapter 파일 있음 + NaN 없음 → ✅

### 점검 포인트
- [ ] exit code 0 (학습 정상 종료)
- [ ] loss가 NaN/Inf 아님 (학습 발산 X)
- [ ] `tests/output/sft_train/checkpoint/adapter_*.safetensors` 존재
- [ ] 메모리 OOM 없이 완주 (Mac 16GB 기준)
- [ ] tokenizer 호환 문제 없음 (Qwen3 chat template 정상 적용)

---

## 4. Phase C — 학습용 본 데이터 만들기 (Stage 3 + 3.5)

### 무엇을 검증하나
- Phase B의 체크포인트(π_ref)로 *on-policy K샘플링*이 정상 동작
- GPT-4o judge가 각 step을 belief-conditional로 라벨링
- Type-1 / Type-2 페어가 schema대로 생성
- flip rate가 측정 가능 (Full 모드만)

### 실행

**Step-DPO 모드 데이터**:
```bash
bash tests/run_pairs.sh step_dpo
```

**Full Step-DPO 모드 데이터** (default — Type-2 페어 포함):
```bash
bash tests/run_pairs.sh full
```

> 같은 빌더 스크립트(`data_pipeline/3_build_pairs.py`)가 *두 모드 모두* Type-1 + Type-2를
> 생성하고, Stage 4 학습 시점에 `disable_type2: true`면 Type-2가 사후 필터된다.
> 따라서 Phase C-1과 C-2의 결과 jsonl은 *실질적으로 같은 분포*. 본 phase는 단지 빌더가
> 두 모드 컨텍스트 모두에서 오류 없이 도는지 확인하는 의미.

### 산출 파일

| 파일 | 내용 |
|---|---|
| `tests/output/pairs_{mode}/preference_pairs.jsonl` | Type-1 + Type-2 페어 |
| `tests/output/pairs_{mode}/flip_stats.json` | flip rate 통계 (Full 모드에서 의미) |
| `tests/output/pairs_{mode}/REPORT.md` | **자동 생성 결과 요약** |

### REPORT.md에 들어가는 것
- 총 페어 수, `pair_type`별 카운트 (step_pair vs belief_flip_pair)
- Type-1의 `reject_type` 분포 (`reject_math` / `reject_persona`)
- 모드별 *실제 학습에 쓰일* 페어 수
- Type-2 flip 매트릭스 (top 10 페르소나 짝)
- flip_stats.json 핵심 지표
- 샘플 페어 각 type 1개씩
- **Pass/Fail Verdict**: Type-1 페어 > 0 (필수), Full 모드라면 Type-2 페어 > 0 (필수)

### 점검 포인트
- [ ] `preference_pairs.jsonl`이 비어 있지 않음
- [ ] Step-DPO 모드: Type-1 페어 ≥ 1 (어떤 페르소나에서 SFT 모델이 실패한 게 있어야)
- [ ] Full 모드: Type-1 페어 + Type-2 페어 모두 ≥ 1
- [ ] **Full 모드 flip rate > 0** (페르소나 신호가 진짜 있는지 — Proposition 3 검증)
- [ ] flip 매트릭스에서 *멀리 떨어진 페르소나 짝* (예: elem_low ↔ high_high)이 더 자주 flip되나? (직관적 기대)
- [ ] Stage 3 import 오류 없음 (vLLM ↔ transformers fallback 정상)

---

## 5. 종합 점검 (Stage 0~3 통합)

세 phase 모두 끝나면 다음 5개 REPORT.md가 생긴다:

```
tests/output/
├── sft_data/REPORT.md          ← Phase A
├── sft_train/REPORT.md         ← Phase B
├── pairs_step_dpo/REPORT.md    ← Phase C-1
└── pairs_full/REPORT.md        ← Phase C-2
```

### 전체 ✅/❌ 결정표

| 항목 | 어느 REPORT에서 확인 | 통과 조건 |
|---|---|---|
| MetaMathQA 다운로드·필터 | `sft_data/REPORT.md` | total rows > 0, augmentation_type 분포 4종 |
| GPT-4o 합성 정상 | `sft_data/REPORT.md` | 페르소나별 행 균등, 평균 풀이 길이 비현실적이지 않음 |
| 페르소나 톤 분기 | `sft_data/REPORT.md` 샘플 행 | 6 페르소나 풀이가 *읽기에 다름* (수동 검토) |
| SFT 학습 완주 | `sft_train/REPORT.md` | exit 0, adapter 파일 있음 |
| 학습 loss 정상 | `sft_train/REPORT.md` | NaN/Inf=0, last loss < first loss |
| 선호 페어 생성 | `pairs_*/REPORT.md` | Type-1 ≥ 1 |
| Type-2 신호 존재 | `pairs_full/REPORT.md` | belief_flip_pair ≥ 1 (Full 모드 필수) |
| flip rate 합리적 | `pairs_full/flip_stats.json` | label_flip_rate_type2 > 0 (이상적으로 5~30%) |

### 일괄 실행 (한 번에)

```bash
export OPENAI_API_KEY=sk-...
bash tests/run_sft_data.sh && \
bash tests/run_sft_train.sh && \
bash tests/run_pairs.sh step_dpo && \
bash tests/run_pairs.sh full
```

총 ~15분, ~$1.3 (Mac M2 16GB 기준 추정).

---

## 6. 자주 발생하는 문제와 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: vllm` | Mac M-series는 정상. transformers fallback 자동 적용. |
| `OPENAI_API_KEY` 환경변수 에러 | `export OPENAI_API_KEY=sk-...` 먼저 |
| Phase B OOM | base 모델을 Qwen3-0.6B로 유지. batch_size·grad_accum은 이미 sanity용으로 최소화. |
| Phase B exit code 1 | training_log.txt 마지막 10라인 확인 — tokenizer mismatch 또는 데이터 형식 오류 |
| Phase C에서 Type-1 페어가 0개 | Phase B의 SFT 모델이 너무 작아서 모든 풀이가 정답일 수 있음 — `K_SAMPLES` 늘리거나 sanity로 인정 |
| Phase C에서 Type-2 페어가 0개 | personas.json의 vocabulary_guide가 약함, 또는 SFT 모델이 페르소나 분기 학습 부족 — personas.json 강화 또는 N_PROBLEMS 늘려서 풀스케일 재시도 |
| GPT-4o rate limit | `data_pipeline/1_synthesize_sft.py`의 `--workers`를 8 → 4로 낮춤 |

---

## 7. 후속 단계

Stage 0~3 sanity 모두 ✅면:
- Stage 4 학습으로 진행 (Step-DPO / Full Step-DPO 각각)
- 풀스케일 학습은 `bash data_pipeline/run_full_pipeline.sh` (Linux+CUDA 권장)
- Stage 4~5 sanity test는 별도 문서로 작성 예정 (`tests/STAGE_4_5_REPORT.md`)
