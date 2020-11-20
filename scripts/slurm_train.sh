#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
WORK_DIR=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

GPUS_PER_NODE=${GPUS_PER_NODE}
GRES_GPU_NUM=${GPUS}
# GRES_GPU_NUM=0
CPUS_PER_TASK=${CPUS_PER_TASK}
# CPUS_PER_TASK=2

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GRES_GPU_NUM} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${ENTRY_PATH} $1 --${ENTRY_KEY_DIR}=$3 ${PY_ARGS} --port=$((RANDOM + 10000)) --launcher=slurm
