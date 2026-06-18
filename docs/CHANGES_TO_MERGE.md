# 머지 시 수동 반영 필요 (구조 divergence)

이 브랜치(`minu123/heldout-eval-results`)는 minu123 작업 사본 기준이라 main과 **디렉토리 구조가 갈라져 있음**(main은 평가 코드를 `evaluation/`로 재구성, minu123은 `data_pipeline/`). 아래는 **자동 복사하지 않은** 변경 — 리뷰 후 main 구조에 맞춰 반영 요망.

## 1. `evaluation/5_evaluate.py` — 평가 judge 모델 선택 추가
main 버전(268줄)과 minu123 버전(218줄)이 다른 갈래라 덮지 않음. 아래 변경만 main의 `evaluation/5_evaluate.py`에 반영:
- judge 모델을 `gpt-4o-mini` 하드코딩 → **`--gpt-model` 인자로 지정 가능**하게.
```python
# 모듈 상단
JUDGE_MODEL = "gpt-4o-mini"
# judge 호출부: model="gpt-4o-mini" → model=JUDGE_MODEL
# main(): parser.add_argument("--gpt-model", default="gpt-4o-mini"); global JUDGE_MODEL; JUDGE_MODEL = args.gpt_model
```
이유: **평가 judge를 데이터 생성(gpt-4o-mini)과 분리**해 gpt-4o로 평가(순환성 제거). 특허 신뢰도용.

## 2. 신규 평가/분석 파일 위치
이 브랜치는 `data_pipeline/`에 뒀으나(minu123 구조), main은 `evaluation/`에 두는 게 일관:
- `eval_belief_flip.py` (Belief-Flip 지표), `aggregate_results.py` (결과표) → main에선 `evaluation/`로 이동 권장.
- 그 경우 슬럼 스크립트(`scripts/*_slurm.sh`) 안의 `data_pipeline/5_evaluate.py`·`data_pipeline/eval_belief_flip.py` 경로도 `evaluation/`로 수정 필요.

## 3. held-out 테스트셋 (gitignore라 미포함)
`data_pipeline/output/sft_test_heldout60.jsonl` 은 `.gitignore`(output/) 대상이라 커밋 안 됨. **재현 방법**: MetaMathQA-40K에서 학습 50문제(seed_problems.jsonl)와 겹치지 않는 60문제 추출(`extract_gt_answer`로 gt 파싱). (기존 `sft_test_eval60.jsonl`은 고유 15문제 + **전부 학습에 포함된 누수**라 평가 부적합 — held-out 신규 생성 필수.)

## 4. 코어 파일 변경 (이 브랜치에 덮음 — PR diff로 리뷰)
- `persona_verifier.py`: StageC(화법) 프롬프트를 "수준 부적합=개념 학년 + CRA 표현방식" 2갈래로 재작성(특허 용어 정렬, 7/7 검증) + judge `seed=42`.
- `data_pipeline/shared_sampling.py`: persona judge **병렬화**(ThreadPool 8) + 행단위 flush + vLLM `seed` + 오염 흡수.
- `data_pipeline/3_build_pairs.py`: math judge 병렬화 + `seed=42`.
- `data_pipeline/4_train_bc_stepdpo.py`: `set_seed` + DataLoader generator(재현성).
- `configs/default.yaml`: `seed: 42`.
→ main의 해당 파일과 갈래가 다르면 **diff 보고 선택 반영**.
