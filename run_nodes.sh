#!/usr/bin/env bash
set -euo pipefail

############################################
# Multi-node PyTorch launcher (no Slurm)
# Works in two modes:
#   A) LAUNCHER mode: Run once on a machine that can SSH to all hosts.
#      It will fan-out to others automatically.
#   B) PER-NODE mode: Run the same file on every node (e.g., via a GUI),
#      no SSH required—the script computes node_rank locally.
############################################

########## USER SETTINGS (edit as needed) ##########
# Hosts: put one hostname or PRIVATE IP per line in hosts.txt
HOSTFILE="${HOSTFILE:-hosts.txt}"

# Project directory (must exist on every node, same path or shared FS)
WORKDIR="${WORKDIR:-$PWD}"

# Python program to run
SCRIPT="${SCRIPT:-overlap_allgather.py}"

# GPUs per node
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Rendezvous TCP port on master
MASTER_PORT="${MASTER_PORT:-29500}"

# Optional: force NIC if needed, e.g. "eth0" (usually leave empty)
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

# Python executable (or full path to your conda env’s python)
PYTHON_BIN="${PYTHON_BIN:-python}"

# If you need to activate a venv/conda on each node, set this:
# e.g., ACTIVATE_CMD='source ~/miniconda3/bin/activate pytorch'
ACTIVATE_CMD="${ACTIVATE_CMD:-}"

# Extra args for your script (if any)
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Logs
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"

# Optional NCCL hygiene
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
# export NCCL_DEBUG=INFO   # uncomment when debugging
####################################################

# Read hosts
if [[ ! -f "$HOSTFILE" ]]; then
  echo "ERROR: hostfile '$HOSTFILE' not found. Create it with one host/IP per line."
  exit 1
fi
mapfile -t HOSTS < <(grep -v '^\s*#' "$HOSTFILE" | grep -v '^\s*$' | awk '!seen[$0]++')
NNODES=${#HOSTS[@]}
if (( NNODES < 1 )); then
  echo "ERROR: no hosts found in $HOSTFILE"
  exit 1
fi

MASTER_ADDR="${HOSTS[0]}"
THIS_HOST="$(hostname -s || hostname || echo unknown)"
# If your GUI gives you the private IP instead of hostname, also try to match IP:
THIS_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "[INFO] Hosts (${NNODES}): ${HOSTS[*]}"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[INFO] NPROC_PER_NODE=$NPROC_PER_NODE  SCRIPT=$SCRIPT  WORKDIR=$WORKDIR"

# Figure out node_rank for this machine (for PER-NODE mode)
NODE_RANK="-1"
for i in "${!HOSTS[@]}"; do
  if [[ "${HOSTS[$i]}" == "$THIS_HOST" || "${HOSTS[$i]}" == "$THIS_IP" ]]; then
    NODE_RANK="$i"
    break
  fi
done

# Common env snippet executed on remote/local before torchrun
common_env() {
  echo "set -euo pipefail"
  echo "cd \"$WORKDIR\""
  [[ -n "$ACTIVATE_CMD" ]] && echo "$ACTIVATE_CMD"
  echo "export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}"
  echo "export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
  [[ -n "$NCCL_SOCKET_IFNAME" ]] && echo "export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
}

run_local_torchrun() {
  local node_rank="$1"
  echo "[RUN] node_rank=$node_rank on $(hostname)"
  ${PYTHON_BIN} -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${node_rank}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    "${SCRIPT}" ${EXTRA_ARGS}
}

############################################
# MODE A: LAUNCHER (we can SSH to all nodes)
############################################
# If we are on MASTER and can SSH to others, fan-out from here.
can_ssh_all=true
for h in "${HOSTS[@]}"; do
  # Skip checking ssh to ourselves
  if [[ "$h" == "$THIS_HOST" || "$h" == "$THIS_IP" ]]; then
    continue
  fi
  if ! timeout 3 bash -lc "ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'echo ok' >/dev/null 2>&1"; then
    can_ssh_all=false
    break
  fi
done

if { [[ "$THIS_HOST" == "$MASTER_ADDR" ]] || [[ "$THIS_IP" == "$MASTER_ADDR" ]]; } && $can_ssh_all; then
  echo "[MODE] LAUNCHER mode via SSH from master ($THIS_HOST)"
  PIDS=()
  trap 'echo; echo "[CLEANUP] stopping..."; for h in "${HOSTS[@]}"; do ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" "pkill -f \"$SCRIPT\" || true; pkill -f torch.distributed.run || true" >/dev/null 2>&1 || true; done; for p in "${PIDS[@]:-}"; do kill "$p" >/dev/null 2>&1 || true; wait "$p" >/dev/null 2>&1 || true; done' INT TERM

  for i in "${!HOSTS[@]}"; do
    h="${HOSTS[$i]}"
    log_out="$LOGDIR/${SCRIPT%.py}_rank${i}.out"
    log_err="$LOGDIR/${SCRIPT%.py}_rank${i}.err"
    echo "[LAUNCH] $h (node_rank=$i) -> $log_out"

    # build remote command
    cmd="$(
      common_env
      echo "echo \"[\\\$(hostname)] starting node_rank=$i\""
      cat <<EOF
${PYTHON_BIN} -m torch.distributed.run \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --node_rank ${i} \
  --rdzv_backend c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  ${SCRIPT} ${EXTRA_ARGS}
EOF
    )"

    # launch via ssh
    ( nohup ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" "bash -lc '$cmd'" >"$log_out" 2>"$log_err" & echo \$! ) &
    PIDS+=($!)
  done

  wait
  echo "[DONE] All nodes finished."
  exit 0
fi

############################################
# MODE B: PER-NODE (run same file on every node)
############################################
if [[ "$NODE_RANK" != "-1" ]]; then
  echo "[MODE] PER-NODE mode on $(hostname)  (node_rank=$NODE_RANK)"
  # Prepare local env
  eval "$(common_env)"
  run_local_torchrun "$NODE_RANK"
  exit 0
fi

echo "ERROR: Could not determine mode."
echo "Run this on the master (with SSH to all hosts) OR run the same script on every host."
echo "Also ensure your hostnames/IPs in '$HOSTFILE' match what 'hostname -s' or 'hostname -I' shows."
exit 1
