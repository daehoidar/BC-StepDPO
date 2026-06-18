#!/usr/bin/env bash
# ovn-strong 캠페인 watchdog. 죽으면 resubmit(skip-if-exists로 resume), 완료 시 자기해제.
set -uo pipefail
PROJ="$HOME/project/Persona-Step-DPO"; cd "$PROJ" || exit 0
LOG="$PROJ/logs/watchdog_ovn.log"; STATE="$PROJ/logs/.watchdog_ovn.state"; MAX=8
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_marker(){ grep -ql "overnight 캠페인 done" "$PROJ"/logs/ovn_strong_*.out 2>/dev/null \
  || [ -f "$PROJ/docs/figures_final/fig_results_table_heldout.png" ]; }
self_off(){ crontab -l 2>/dev/null | grep -v 'watchdog_ovn.sh' | crontab - 2>/dev/null; log "watchdog 자기해제"; }

QS="$(squeue -u "$USER" -n ovn-strong -h -o '%T' 2>/dev/null)"
if [ -n "$QS" ]; then log "OK: ovn-strong 큐에 있음($QS)"; exit 0; fi
if done_marker; then log "SUCCESS: 완료 감지"; self_off; exit 0; fi
R=0; [ -f "$STATE" ] && R=$(grep -oE '[0-9]+' "$STATE" | head -1 || echo 0); R=${R:-0}
if [ "$R" -ge "$MAX" ]; then log "STOP: 재시도 상한($MAX)"; self_off; exit 0; fi
export OPENAI_API_KEY="$(cat "$PROJ/.openai_key_fallback" 2>/dev/null)"
export OPENAI_API_KEY_FALLBACK="$(cat "$PROJ/.openai_key" 2>/dev/null)"
NID=$(sbatch --parsable "$PROJ/scripts/overnight_strengthen_slurm.sh" 2>>"$LOG"); RC=$?
R=$((R+1)); echo "RETRIES=$R" > "$STATE"
[ $RC -eq 0 ] && log "RESUBMIT #$R: $NID (resume)" || log "RESUBMIT #$R 실패 rc=$RC"
exit 0
