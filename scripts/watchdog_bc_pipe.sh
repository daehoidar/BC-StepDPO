#!/usr/bin/env bash
# watchdog_bc_pipe.sh — bc-pipe SLURM 잡 무인 자동수정 watchdog.
#
# cron이 주기적으로 호출. 동작:
#   - bc-pipe 잡이 큐(R/PD)에 있으면 → 정상, 아무것도 안 함.
#   - 큐에 없는데 파이프라인이 완료됐으면 → 성공 로그 남기고 cron 자기 해제.
#   - 큐에 없는데 미완료면 → 죽은 것. 마지막 에러를 로그에 남기고,
#     재시도 상한 미만이면 동일 파라미터로 resubmit(샘플링은 resume됨).
#
# 모든 동작은 logs/watchdog.log 에 타임스탬프와 함께 기록된다.
set -uo pipefail

PROJ="$HOME/project/Persona-Step-DPO"
cd "$PROJ" || exit 0

LOG="$PROJ/logs/watchdog.log"
STATE="$PROJ/logs/.watchdog.state"     # RETRIES=<n>
MAX_RETRIES=5
JOBNAME="bc-pipe"
DONE_DIR="$PROJ/checkpoints/bc_stepdpo"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 완료 판정: 학습 산출물(adapter)이 존재하거나 .out 에 done 마커.
pipeline_done() {
    [ -f "$DONE_DIR/adapter_model.safetensors" ] && return 0
    ls "$DONE_DIR"/adapter_model*.safetensors >/dev/null 2>&1 && return 0
    grep -ql "=== done ===" "$PROJ"/logs/bc_pipe_*.out 2>/dev/null && return 0
    return 1
}

self_disable() {   # 자기 crontab 라인 제거
    crontab -l 2>/dev/null | grep -v 'watchdog_bc_pipe.sh' | crontab - 2>/dev/null
    log "watchdog cron 자기 해제 완료."
}

# 큐에 bc-pipe 잡이 있나? (R 또는 PD)
QSTAT="$(squeue -u "$USER" -n "$JOBNAME" -h -o '%T' 2>/dev/null)"
if [ -n "$QSTAT" ]; then
    log "OK: bc-pipe 잡 큐에 있음 (state=$(echo "$QSTAT" | tr '\n' ',')). 대기."
    exit 0
fi

# 큐에 없음 → 완료/실패 분기
if pipeline_done; then
    log "SUCCESS: 파이프라인 완료 감지 (checkpoints/bc_stepdpo). 더 이상 감시 불필요."
    self_disable
    exit 0
fi

# 미완료 + 큐에 없음 = 죽음. 재시도 카운트 확인.
RETRIES=0
[ -f "$STATE" ] && RETRIES=$(grep -oE '[0-9]+' "$STATE" | head -1 || echo 0)
RETRIES=${RETRIES:-0}

LASTERR="$(ls -t "$PROJ"/logs/bc_pipe_*.err 2>/dev/null | head -1)"
ERRTAIL="$(tail -8 "$LASTERR" 2>/dev/null | tr '\n' '|')"
log "FAIL: bc-pipe 큐에 없고 미완료. 마지막 에러파일=$LASTERR | tail: $ERRTAIL"

if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
    log "STOP: 재시도 상한($MAX_RETRIES) 도달. 자동 재제출 중단. 사람이 확인 필요."
    self_disable
    exit 0
fi

# resubmit (샘플링 resume, 동일 파라미터). API 키는 파일에서.
# 개인 키(.openai_key_fallback)가 빠르고 안정적이라 1순위로 사용, 팀 키는 백업.
export OPENAI_API_KEY="$(cat "$PROJ/.openai_key_fallback" 2>/dev/null)"
export OPENAI_API_KEY_FALLBACK="$(cat "$PROJ/.openai_key" 2>/dev/null)"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
NEWID=$(MAX_ROWS=0 K_SAMPLES=16 CONFIG=configs/default.yaml OUT=checkpoints/bc_stepdpo \
        ADAPTER=checkpoints/sft_qwen3_1.7b_eos \
        sbatch --parsable scripts/bc_stepdpo_pipeline_slurm.sh 2>>"$LOG")
RC=$?
RETRIES=$((RETRIES+1))
echo "RETRIES=$RETRIES" > "$STATE"
if [ $RC -eq 0 ] && [ -n "$NEWID" ]; then
    log "RESUBMIT #$RETRIES: 새 잡 $NEWID 제출됨 (resume)."
else
    log "RESUBMIT #$RETRIES 실패 (rc=$RC). 다음 cron에서 재시도."
fi
exit 0
