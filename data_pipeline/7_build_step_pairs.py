"""정답 재샘플링 결과로 step_pair 데이터를 빌드한다.

입력: data_pipeline/output/correction_inputs.jsonl + corrections/*.json
출력: data_pipeline/output/step_pairs.jsonl

각 행 형식:
    {"type": "step_pair", "persona": "초등-하위권", "problem_id": "...",
     "prompt": "...",
     "prefix": "Let's think step by step.\\nStep 1: ...\\nStep 2:",
     "chosen": " 정답 첫 스텝", "rejected": " 오답 첫 스텝",
     "gt_answer": "...", "cut_point": 2,
     # --- 데이터 품질 메타데이터 (3→4→5번에서 전파) ---
     "n_samples": 8,           # rep
     "n_failures": 3,          # result=False 개수
     "attempt_index": 1,       # rejected가 N샘플 중 0-indexed 몇 번째에서 왔는지
     "failure_rate": 0.375,    # n_failures / n_samples
     "sampling_config": {"temperature": 0.7, "top_p": 0.95, "seed": 42}}

TODO: Step-DPO 레포의 generate_dataset.py를 베이스로 작성.
chosen/rejected는 첫 스텝만 비교 (split('\\nStep ')[0]).
"""
