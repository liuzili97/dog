#!/usr/bin/bash

set -x

CONFIG=$1
GRES_GPU_NUM=$2
WORK_DIR=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

CPUS_PER_TASK=8

srun -p ${PARTITION} \
    -w ${NODE_NAMES} \
    --job-name=${JOB_NAME} \
    --mem-per-cpu=6G \
    --qos=gpu \
    --gres=gpu:${GRES_GPU_NUM} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${ENTRY_PATH} $1 $2 --${ENTRY_KEY_DIR}=$3 ${PY_ARGS} --slurm-mode=True

