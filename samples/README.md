# BC-StepDPO 학습 데이터 합성 — 한 문제 walkthrough

[data_pipeline/](../data_pipeline/)이 완전히 구현되었을 때 *한 문제·6 페르소나*에 대해 산출되는 모든 파일을 [make_example.py](make_example.py)가 GPT-4o 호출로 재현한다. 팀 리뷰 + 페이퍼 supplementary 후보.

> **설정 요약**
> - 시드 데이터: [MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K) (GSM8K + MATH 증강). 파이프라인은 `type ∈ GSM_*`만 필터 + `query` dedupe로 unique question만 사용. 자세한 사실 검증은 [0_seed_problems.py](../data_pipeline/0_seed_problems.py) docstring.
> - 프롬프트 언어: **영어 입출력**. system prompt body는 한국 2022 개정 교육과정 텍스트라 한국어로 유지하되 GPT-4o는 영어로 응답 강제.
> - SFT base 모델: **Qwen3-4B-Instruct** (학습 산출물 `outputs/pi_sft_qwen3_4b/`)
> - make_example.py의 SFT proxy: `gpt-4o-mini` (실제 모델 학습 전 stand-in). 본격 실험에선 `qwen3:4b`(Ollama) 또는 학습 결과로 교체.

## 실행

```bash
export OPENAI_API_KEY=sk-...
python samples/make_example.py
```

| 옵션 | 의미 |
|---|---|
| `--question`, `--gt-answer`, `--problem-id` | 다른 문제로 갈아끼우기 |
| `--n-samples N` | SFT proxy 샘플 수 (기본 8). `rep`에 해당 |
| `--cut-k K` | anchored continuation의 cut point. 기본 2 |
| `--skip-sft` | 기존 `sft_data.jsonl` 재사용 |
| `--skip-type1` / `--skip-type2` | 한쪽만 빠르게 다시 돌릴 때 |

비용 자릿수: **~$0.50/문제** (모델 가격 변동 가능).

## 산출 파일

### 학습에 직접 소비되는 파일

| 파일 | 소비처 | 내용 |
|---|---|---|
| `sft_data.jsonl` | [2_run_sft.sh](../data_pipeline/2_run_sft.sh) (SFT training) | 페르소나 6종 × 정답 풀이 |
| `train.jsonl` | BC-StepDPO training loop | `step_pairs` + `belief_pairs` merge. 모든 행이 단일 손실로 처리 |

### 중간 산출물 (검증·디버그용)

| 파일 | 단계 | 내용 |
|---|---|---|
| `seed_problems.jsonl` | 0 | 입력 시드 (6 페르소나 복제) |
| `predictions.jsonl` | 3 | SFT proxy N샘플링 결과 + 정/오 라벨 |
| `step_pairs.jsonl` | 7 | Type-1: math 축 페어 (locate → rectify) |
| `alt_steps.jsonl` | 1c | anchored continuation 원본 + math judge 결과 |
| `belief_pairs.jsonl` | 8 | Type-2: Step 1 cross-persona + anchored (judge 통과분) |

## 파이프라인 매핑

| make_example.py 단계 | 실제 파이프라인 | 사용 모델 | 산출 |
|---|---|---|---|
| [1/6] SFT 합성 | [1_synthesize_sft.py](../data_pipeline/1_synthesize_sft.py) | gpt-4o (T=0.3) | sft_data.jsonl |
| [2/6] belief_pair Step 1 | [8_build_belief_pairs.py](../data_pipeline/8_build_belief_pairs.py) (Step 1 부분) | — (slice only) | belief_pairs.jsonl |
| [3/6] anchored + math judge | 1c (신규) + 1d filter | gpt-4o (T=0.4) + gpt-4o judge | alt_steps.jsonl |
| [4/6] SFT proxy 샘플링 | [3_collect_errors.sh](../data_pipeline/3_collect_errors.sh) | **gpt-4o-mini** stand-in (실제는 Qwen3-4B 학습 후 그 모델 사용, T=1.0) | predictions.jsonl |
| [5/6] locate + rectify | [4](../data_pipeline/4_locate_error.py) + [5](../data_pipeline/5_prepare_correction.py) + [6](../data_pipeline/6_rectify.sh) + [7번](../data_pipeline/7_build_step_pairs.py) | gpt-4o (judge·rectify) | step_pairs.jsonl |
| [6/6] merge | [9_merge.py](../data_pipeline/9_merge.py) | — | train.jsonl |

> ⚠️ **SFT 모델 proxy 주의사항**: 학습 대상 SFT 모델(Qwen3-4B 기반)이 아직 준비되지 않아서 [4/6] 단계는 `gpt-4o-mini`를 stand-in으로 쓴다. 실제 SFT 모델이 준비되면 `SFT_PROXY_MODEL`을 갈아끼우면 됨. *쉬운* 문제는 gpt-4o-mini가 거의 안 틀리므로 Type-1 페어가 드물거나 0개일 수 있음 — 실제 Qwen3-4B 기반 SFT 모델에선 failure_rate가 더 높을 것.

## Quality gate

원본 Step-DPO 분업을 그대로 따라 **SFT 데이터에는 final-answer 검사만** 적용한다 (step-level judge 없음). 중간 noise는 DPO loop가 보정.

| 단계 | 검사 | 실패 시 |
|---|---|---|
| SFT 합성 | `\boxed{}`와 `gt_answer` 일치 (`answer_correct`) | 행 유지하되 Type-2 페어 구성에서 제외 |
| Type-1 locate | GPT-4o judge가 first-error step 번호 산출 | 그 페르소나 step_pair 스킵 |
| Type-1 rectify | rectify된 step에 step-level math judge 재확인 | judge fail 시 페어 drop |
| Type-2 anchored | `chosen_regen`·`alt_step` 양쪽 모두 step-level math judge | 한쪽이라도 fail이면 페어 drop |

## Type-1 vs Type-2 (손실 axis 분리)

| | Type-1 `step_pair` | Type-2 `belief_pair` |
|---|---|---|
| 차이 축 | math correctness | belief / persona |
| chosen | rectified step (math ✓) | target 페르소나 step (math ✓) |
| rejected | SFT proxy 자연 오답 (math ✗) | alt 페르소나 step (math ✓ + belief ✗) |
| 페르소나 톤 | chosen·rejected *동일* (target) | chosen·rejected *다름* (target vs alt) |
| 통계 메타데이터 | `n_samples`, `n_failures`, `attempt_index`, `failure_rate`, `sampling_config` | `math_status` (judge 결과) |
| 한 문제당 페어 수 | 페르소나당 최대 1 (실패 있을 때만) | Step 1 cross 최대 30 + anchored 최대 6 |

두 페어 타입 모두 **단일 BC-StepDPO 손실**로 처리. `type` 필드는 통계·디버깅용 메타데이터일 뿐 학습 분기 신호가 아니다.

## Label flip 증거 (Proposition 3)

Step 1 cross-persona 페어가 **양방향**으로 생성되므로, *같은 step 텍스트*가 한 belief에서 chosen이고 다른 belief에서 rejected가 되는 flip이 자연 발생한다. 페이퍼/특허에서 (A7) belief-dependent reward의 empirical 증거로 사용.

집계 예시 (`belief_pairs.jsonl`에서 직접 셀 수 있음):
```python
from collections import defaultdict
import json
chosen, rejected = defaultdict(set), defaultdict(set)
for row in map(json.loads, open("samples/belief_pairs.jsonl")):
    key = (row["problem_id"], row.get("prefix", ""), row["chosen"])
    chosen[key].add(row["persona"])
    key_r = (row["problem_id"], row.get("prefix", ""), row["rejected"])
    rejected[key_r].add(row["persona"])
flips = sum(1 for k in chosen if k in rejected and chosen[k] & rejected[k] == set())
print(f"flip instances: {flips}")
```

## 팀 검토 체크리스트

- [ ] **SFT 톤 분리**: 6 페르소나의 Step 1이 충분히 구별되나? 인접 페르소나(중등-상위권 vs 고등-하위권) 차이가 명확한가?
- [ ] **SFT 정확도**: `sft_data.jsonl`의 모든 행에서 `answer_correct: true`인가?
- [ ] **Type-1 톤 일관성**: `step_pairs.jsonl`의 chosen/rejected가 같은 페르소나 톤을 유지하나? rectified step이 페르소나 어휘를 그대로 쓰나?
- [ ] **Type-1 failure_rate 분포**: 페르소나별 `n_failures`가 합리적인가? (쉬운 문제는 0이 정상)
- [ ] **Type-2 anchored 톤 분기**: rejected가 prefix 톤에 끌려 흐려지지 않았나? `alt_step`이 정말 alt 페르소나 색깔인지
- [ ] **Type-2 chosen 재생성 효과**: `alt_steps.jsonl`의 `chosen_regen`이 원본 `sft_data.jsonl` Step k와 *살짝* 달라야 자연스러움 (single-variable manipulation 부수효과)
- [ ] **math judge 합격률**: anchored 페어 중 양쪽 다 `pass`인 비율. 너무 낮으면 ANCHORED_INSTRUCTION 튜닝 필요
- [ ] **flip rate**: Step 1 cross-persona에서 flip이 직관적으로 "페르소나 따라 다르게 평가될 만하다"고 느껴지나?

## 한계 (페이퍼 §6 L1–L4)

- **(L1)** 단일 라벨: math 오류와 persona 미스매치를 모두 "rejected"로 통합. 페어 타입 분리로 학습 시 disentangle.
- **(L2)** Judge 의존성: math correctness 판정이 GPT-4o judge에 의존. judge bias가 데이터에 전파될 수 있음.
- **(L3)** Belief 외생성: belief를 "persona system prompt $\phi_b$가 인코딩하는 발화 정책"으로 *operationally* 정의. → 강점: 교사·제품이 $\phi_b$를 후처리로 수정해 belief를 갱신할 수 있는 *조작 가능한 변수*. 한계: 인간 belief로의 일반화는 future work.
- **(L4)** Type-2엔 first-error localization 미적용. persona drift는 step 간 causal propagation이 약해 단일 cut point + 무작위 alt persona 샘플링으로 둠.
