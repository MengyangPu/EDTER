#!/usr/bin/env bash

PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    $(dirname "$0")/test_local.py --launcher pytorch ${@:1}
