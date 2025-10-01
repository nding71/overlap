#!/usr/bin/env bash
set -euo pipefail

############################################
# Multi-node launcher via shared filesystem (no IPs/ports; no GUI env needed)
# Run THIS SAME FILE on EVERY node in your job.
# It auto-assigns node_rank by writing to a shared folder.
############################################

########## CONFIG — edit these 3 lines to match your setup ##########
NNODES=2                                  # total number of nodes in your job
WORKDIR="/shared/project/overlap"         # folder that ALL nodes can see
RDZV_DIR="/shared/rdzv"                   # shared, writable folder for rendezvous
########################################################################

# Script & Python (adjust if needed)
SCRIPT="overlap_allgather.py"
PYTHON_BIN="python"
ACTIVATE_CMD=""     # e.g.: ACTIVATE_CMD='source ~/miniconda3/bin/activate pt'
LOGDIR="${WORKDIR}/logs"

# Autodetect GPUs per node (fallback to 8)
detect_nproc() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  elif command -v rocminfo >/dev/null 2>&1; then
    # crude count of GPUs on ROCm
    rocminfo | grep -c "Compute Unit" || echo 8
  else
    echo 8
  fi
}
NPROC_PER_NODE="$(detect_nproc)"

# Niceties
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO   # uncomment for debugging

mkdir -p "$LOGDIR" "$RDZV_DIR" "$RDZV_DIR/nodes"
cd "$WORKDIR"
[[ -n "$ACTIVATE_CMD" ]] && eval "$ACTIVATE_CMD"

# Assign a stable node_rank using a shared file + lock
NODE_FILE="$RDZV_DIR/nodes/list.txt"
LOCK_FILE="$RDZV_DIR/nodes/lockfile"
hostname_short="$(hostname -s || hostname || echo node)"
this_id="$hostname_short-$$-$(date +%s%N)"

exec 9>"$LOCK_FILE"
flock 9
# write once
if ! grep -qxF "$this_id" "$NODE_FILE" 2>/dev/null; then
  echo "$this_id" >> "$NODE_FILE"
fi
# compute node_rank = (line number - 1)
node_rank=$(( $(grep -n -x "$this_id" "$NODE_FILE" | cut -d: -f1) - 1 ))
flock -u 9

if (( node_rank < 0 )); then
  echo "[FATAL] Could not determine node_rank"; exit 1
fi

world_size=$(( NNODES * NPROC_PER_NODE ))
INIT_METHOD="file://$RDZV_DIR/pg"

echo "[INFO] node_rank=$node_rank  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$world_size"
echo "[INFO] WORKDIR=$WORKDIR  RDZV_DIR=$RDZV_DIR"
echo "[INFO] Launching $NPROC_PER_NODE processes on $(hostname)"

# Launch one process per GPU with required env for file rendezvous
pids=()
for (( local_rank=0; local_rank<NPROC_PER_NODE; local_rank++ )); do
  global_rank=$(( node_rank * NPROC_PER_NODE + local_rank ))
  log_base="$LOGDIR/${SCRIPT%.py}_n${node_rank}_l${local_rank}"

  echo "[LAUNCH] node_rank=$node_rank local_rank=$local_rank global_rank=$global_rank -> $log_base.out"

  RANK="$global_rank" WORLD_SIZE="$world_size" LOCAL_RANK="$local_rank" INIT_METHOD="$INIT_METHOD" \
  "$PYTHON_BIN" "$SCRIPT" >"$log_base.out" 2>"$log_base.err" &

  pids+=($!)
done

# Wait for all local procs
code=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then code=1; fi
done
exit $code
