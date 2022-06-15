#!/usr/bin/env bash

GLOBALCONFIG=$1
CONFIG=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_local.py $GLOBALCONFIG $CONFIG --launcher pytorch ${@:4}
