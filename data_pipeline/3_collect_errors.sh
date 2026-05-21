#!/usr/bin/env bash
# 참조 모델 pi_SFT로 페르소나 조건부 풀이를 대량 샘플링하여 정/오 라벨링한다.
# Step-DPO 원본 step1.sh의 페르소나 변형.
#
# 입력: SFT된 모델 (outputs/pi_sft_qwen3_0.6b)
#       문제 jsonl (페르소나 6종 x rep회로 펼침; 권장 rep>=4)
# 출력: data_pipeline/output/predictions/*.json
#
# 한 행 = 한 (problem_id, persona, attempt) 샘플. 필드:
#   problem_id, persona, question, gt_answer, model_solution, result(bool),
#   n_samples       : 이 (problem_id, persona)에 대해 총 몇 번 샘플링했나 (= rep)
#   n_failures      : 그 중 result=False인 샘플 수
#   attempt_index   : 이 행이 n_samples 중 0-indexed 몇 번째 샘플인지
#   failure_rate    : n_failures / n_samples (derived; 페이퍼 표/curriculum sorting용)
#   sampling_config : {"temperature": ..., "top_p": ..., "seed": ...}
#
# 위 메타데이터는 4→5→6→7번까지 그대로 전파되어, 최종 step_pair 한 행이
# "어떤 샘플링 조건에서 N번 중 몇 번째로 실패한 출력에서 만들어졌는지"를 증빙한다.
#
# TODO: eval_math_persona.py 작성 후 vLLM 추론 호출.

set -e
echo "TODO: implement error collection"
