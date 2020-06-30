#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
WORK_DIR=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--$ENTRY_KEY_DIR=$3"}

srun -p ${PARTITION} \
    -w node18 \
    --job-name=${JOB_NAME} \
    --gres=gpu:4 \
    --ntasks=${GPUS} \
    --ntasks-per-node=4 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${ENTRY_PATH} $1 ${PY_ARGS}
