#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${PY_ARGS:-""}
GPU_NUM=$[($(echo ${GPUS} | wc -L)+1)/2]

CUDA_VISIBLE_DEVICES=${GPUS} python ${ENTRY_PATH} $1 ${PY_ARGS} --${ENTRY_KEY_DIR}=${WORK_DIR} --num-gpus=${GPU_NUM}
