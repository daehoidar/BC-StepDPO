"""오류 스텝 직전까지의 prefix를 추출하여 재샘플링용 입력 jsonl을 만든다.

입력: data_pipeline/output/located_errors/*.json
출력: data_pipeline/output/correction_inputs.jsonl

3번에서 시작된 메타데이터(n_samples, n_failures, attempt_index, failure_rate,
sampling_config)와 4번의 first_error_step_idx를 한 줄도 빠뜨리지 말고 보존한다.
persona 필드도 보존.
"""
