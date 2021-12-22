#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${PY_ARGS:-""}

python ${ENTRY_PATH} $1 $2 ${PY_ARGS} --${ENTRY_KEY_DIR}=${WORK_DIR}
