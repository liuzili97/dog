#!/usr/bin/env bash

set -x

CONFIG=$1
GRES_GPU_NUM=$2
WORK_DIR=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

# CPUS_PER_TASK=${CPUS_PER_TASK}
CPUS_PER_TASK=4

srun -p ${PARTITION} \
    -w ${NODE_NAMES} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GRES_GPU_NUM} \
    --ntasks=${NTASKS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${ENTRY_PATH} $1 --${ENTRY_KEY_DIR}=$3 ${PY_ARGS} --port=$((RANDOM + 10000)) --launcher=slurm --seed 123 --deterministic
