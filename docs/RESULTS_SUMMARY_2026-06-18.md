# BC-StepDPO 평가 결과 정리 (held-out 재평가) — 2026-06-18

> 팀 공유용. 기존 결과표의 **데이터 누수·순환 judge·작은 표본**을 모두 바로잡은 신뢰 가능한 재평가.

## 1. 무엇을 고쳤나 (기존 표의 문제 → 수정)
| 문제 | 기존 | 수정 |
|---|---|---|
| **데이터 누수** | test 15문제가 **전부 학습 samples에 포함**(held-out 아님) | MetaMathQA에서 학습과 **겹침 0**인 **held-out 60문제** 신규 생성 (`sft_test_heldout60.jsonl`) |
| **순환 judge** | 평가 judge = 데이터 생성 judge = `gpt-4o-mini` | 평가를 **`gpt-4o`**(독립·강함)로 |
| **작은 표본** | belief-flip **n=15** (문제 1~2개 = 노이즈) | **n=60** |
| **Type-2 부족** | belief-flip 페어 **31개** | 제약 완화로 **100개**(full_aug) |

평가 디렉토리 `eval_ho/`, 표 `docs/figures_final/fig_results_table_heldout.png`.

## 2. 결과 표 (held-out · gpt-4o · n=60)

| Model | Final Acc | Step Acc | **Persona Cons** | Format | **Belief-Flip** |
|---|---|---|---|---|---|
| SFT (Baseline) | 73.9 | 91.5 | 79.5 | 98.1 | 8.3 |
| Vanilla DPO | **76.4** | **92.9** | 80.1 | 98.6 | 10.0 |
| Step-DPO | 72.2 | 90.9 | 79.1 | **98.9** | 10.0 |
| Full-Step-DPO (PRM) | — | — | — | — | — |
| BC-StepDPO (Type-1) | 72.2 | 89.9 | 81.4 | 98.1 | **11.7** |
| Full BC-StepDPO (T1+T2) | 72.8 | 91.6 | 81.7 | 98.9 | 10.0 |
| Full BC-StepDPO (T2 aug, 100) | 68.9 | 90.4 | **83.1** | 98.1 | 6.7 |

belief-flip 분해(참고): 모든 모델 `lo_ok`(저학년 풀이가 저학년에 적합) **25~40%로 낮음** = 모델이 elem_low 수준 풀이를 잘 못 만드는 게 병목.

## 3. 핵심 발견

### ① 기존 숫자는 누수로 부풀려져 있었음
- **Final Acc: 90.3 → 73.9** (SFT). 누수가 **~15~17점** 부풀림. Persona Cons도 gpt-4o로 94→~80.
- → 기존 표는 폐기, **held-out 표가 정확.**

### ② Persona Cons: BC-StepDPO가 소폭 우위 ✓
- baseline 79~80 < **BC-StepDPO 81.4~83.1**. **Type-2 증량(full_aug=83.1)이 최고.**
- 여기선 Type-2 증량이 **도움**(81.7→83.1).

### ③ Belief-Flip(핵심 novelty): 입증 실패 ✗
- BC-StepDPO가 baseline보다 약간 위(11.7/10 vs 8.3~10)지만 차이가 작고(n=60에서 1~3문제),
- **Type-2를 31→100으로 늘린 full_aug가 오히려 6.7로 최악** — 가설과 정반대.
- 추정: 완화 생성한 Type-2 품질 저하 / 과도한 드리프트(full_aug는 Final Acc 68.9로 최저).

### ④ Full-Step-DPO(PRM): CUDA OOM으로 미완 (복구 가능)
- 3a(MC rollout) 성공 → **3b PRM 학습이 OOM**(기본 batch=8이 A10 22GB엔 큼). `batch_size↓ + grad_accum↑`로 재시도하면 채울 수 있음.

## 4. 정직한 해석 (구조적 원인)
- **persona 적합성은 대부분 SFT 단계에서 학습됨** (SFT 데이터가 페르소나별 합성 풀이 → SFT base Persona Cons 79.5). 모든 DPO 변형이 이 base에서 출발.
- 그래서 belief-conditioning(belief 토큰 + Type-2)은 **이미 잘하는 baseline 위의 작은 추가** → 한계효용이 작음. Step-DPO baseline도 persona 페어를 학습.
- 결과: **Persona Cons는 소폭 win이지만, 핵심 주장인 Belief-Flip은 데이터로 뒷받침 안 됨.**

## 5. 다음 단계 옵션
1. **Full-Step-DPO OOM 수정** → 6번째 행 채우기 (배치 축소).
2. **Belief-Flip 병목 분석**: `lo_ok` 낮음(저학년 풀이 품질) — SFT 단계 저학년 데이터/프롬프트 보강.
3. **실험설계 보강**: persona를 약하게 한 SFT base와 비교해 belief-conditioning **순효과를 분리**. (현재는 SFT가 persona를 다 해서 novelty가 묻힘.)
4. **주장 재구성**: Belief-Flip 대신 **Persona Cons 중심**으로(소폭이지만 일관된 우위 + Type-2 aug 효과).
5. Type-2 **품질** 개선(완화 생성이 역효과 → cross-belief 판정 강화 / gpt-4o로).

## 6. 재현 정보
- held-out test: `data_pipeline/output/sft_test_heldout60.jsonl` (MetaMathQA, seed 무관 추출, 학습 겹침 0)
- 캠페인: `scripts/overnight_strengthen_slurm.sh` (P1 재평가 / P2 Type-2 augment+재학습 / P3 Full-Step-DPO / P4 집계)
- Type-2 증량: `data_pipeline/augment_type2.py` (후보 5·max 20·병렬)
- 평가 judge 지정: `5_evaluate.py --gpt-model gpt-4o`, `eval_belief_flip.py --gpt-model gpt-4o`
- 모델: base=`checkpoints/sft_qwen3_1.7b_eos_merged`, full=`bc_stepdpo_v3`, full_aug=`full_aug` (config `bc_retrain_v3.yaml`, seed 42)
- 전 단계 seed 42 고정(재현 가능).
