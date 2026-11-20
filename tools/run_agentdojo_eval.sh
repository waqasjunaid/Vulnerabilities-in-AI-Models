#!/usr/bin/env bash
# Frozen AgentDojo evaluation for the guardian. One call = one mode on all suites.
#
#   tools/run_agentdojo_eval.sh baseline runs/agentdojo/final_baseline
#   tools/run_agentdojo_eval.sh guarded  runs/agentdojo/final_guarded
#
# Each call runs two passes per suite:
#   attack : every user task x every injection task (AgentDojo --attack)
#   clean  : every user task with no injection (needed for clean utility)
#
# Environment overrides:
#   PROJECT   guardian_lab root         (default: current directory)
#   MODEL_ID  Ollama model              (default: qwen2.5:14b)
#   SUITES    space-separated suites    (default: "workspace banking slack travel")
#   PHASES    "attack clean"            (default: both)
#   ATTACK    AgentDojo attack name     (default: injecagent)
#   THRESHOLD guardian threshold        (default: 0.75; do not change after seeing results)
#   GUARDIAN_VERSION  v1 (default) or v2 (argument-grounded, needs trust_v2.py in PROJECT)
#   EXTRA     extra benchmark flags, e.g. EXTRA="--user-task user_task_1 --user-task user_task_2"
#
# Safe to resume: AgentDojo skips cases already present in the log directory
# (no --force-rerun), so re-running with the same OUT continues where it stopped.
set -u

MODE="${1:-}"
OUT="${2:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "guarded" || -z "$OUT" ]]; then
  echo "usage: $0 baseline|guarded OUT_DIR" >&2
  exit 2
fi

PROJECT="${PROJECT:-$PWD}"
case "$OUT" in /*) ;; *) OUT="$PROJECT/$OUT" ;; esac
AGENTDOJO="$PROJECT/Agentdojo"
MODEL_ID="${MODEL_ID:-qwen2.5:14b}"
SUITES="${SUITES:-workspace banking slack travel}"
PHASES="${PHASES:-attack clean}"
ATTACK="${ATTACK:-injecagent}"
THRESHOLD="${THRESHOLD:-0.75}"
EXTRA="${EXTRA:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$AGENTDOJO/src/agentdojo/agent_pipeline/agent_pipeline.py"

echo "== preflight: syntax =="
python -m py_compile "$PIPE" "$PROJECT/trust_model.py" "$PROJECT/agentdojo_guardian.py" || exit 1
LEAK_FILES=("$PIPE" "$PROJECT/agentdojo_guardian.py" "$PROJECT/trust_model.py")
if [[ -f "$PROJECT/trust_v2.py" ]]; then
  python -m py_compile "$PROJECT/trust_v2.py" || exit 1
  LEAK_FILES+=("$PROJECT/trust_v2.py")
fi

echo "== preflight: benchmark-specific content =="
python "$HERE/check_leakage.py" "${LEAK_FILES[@]}" || exit 1

if [[ "$MODE" == "guarded" ]]; then
  export GUARDIAN_ENABLED=1
else
  export GUARDIAN_ENABLED=0
fi
export GUARDIAN_PROJECT_ROOT="$PROJECT"
export TRUST_DETECTOR="${TRUST_DETECTOR:-heuristic}"
export GUARDIAN_THRESHOLD="$THRESHOLD"
export GUARDIAN_RESPONSE_MIN_SCORE="${GUARDIAN_RESPONSE_MIN_SCORE:-0.50}"
export GUARDIAN_VERSION="${GUARDIAN_VERSION:-v1}"
export PYTHONPATH="$AGENTDOJO/src:$PROJECT:${PYTHONPATH:-}"
export OPENAI_COMPATIBLE_BASE_URL="${OPENAI_COMPATIBLE_BASE_URL:-http://127.0.0.1:11434/v1}"
export OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:-ollama}"

mkdir -p "$OUT"
{
  echo "mode=$MODE model=$MODEL_ID attack=$ATTACK threshold=$THRESHOLD suites=$SUITES phases=$PHASES"
  echo "extra=$EXTRA"
  echo "started=$(date -Is)"
  echo "pipeline_sha256=$(sha256sum "$PIPE" | cut -d' ' -f1)"
  echo "trust_model_sha256=$(sha256sum "$PROJECT/trust_model.py" | cut -d' ' -f1)"
  echo "adapter_sha256=$(sha256sum "$PROJECT/agentdojo_guardian.py" | cut -d' ' -f1)"
  echo "guardian_version=$GUARDIAN_VERSION response_min_score=$GUARDIAN_RESPONSE_MIN_SCORE largest_file_policy=${GUARDIAN_LARGEST_FILE_POLICY:-0}"
  [[ -f "$PROJECT/trust_v2.py" ]] && echo "trust_v2_sha256=$(sha256sum "$PROJECT/trust_v2.py" | cut -d' ' -f1)"
  [[ -f "$AGENTDOJO/src/agentdojo/agent_pipeline/llms/openai_llm.py" ]] && echo "openai_llm_sha256=$(sha256sum "$AGENTDOJO/src/agentdojo/agent_pipeline/llms/openai_llm.py" | cut -d' ' -f1)"
} >> "$OUT/RUN_INFO.txt"
cat "$OUT/RUN_INFO.txt"

cd "$AGENTDOJO" || exit 1
STATUS=0
for PHASE in $PHASES; do
  for SUITE in $SUITES; do
    echo; echo "========== $MODE / $PHASE / $SUITE =========="
    ATTACK_FLAG=()
    [[ "$PHASE" == "attack" ]] && ATTACK_FLAG=(--attack "$ATTACK")
    # shellcheck disable=SC2086
    python -m agentdojo.scripts.benchmark \
      --model OPENAI_COMPATIBLE \
      --model-id "$MODEL_ID" \
      --benchmark-version v1.2.2 \
      "${ATTACK_FLAG[@]}" \
      --suite "$SUITE" \
      $EXTRA \
      --logdir "$OUT/$PHASE/$SUITE" 2>&1 | tee -a "$OUT/${PHASE}_${SUITE}_console.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "$PHASE/$SUITE failed"; STATUS=1; }
  done
done
echo "finished=$(date -Is)" >> "$OUT/RUN_INFO.txt"
exit $STATUS
