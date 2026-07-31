#!/bin/bash
#GQ gpus=1 name=stress

# Directives above replace the inline flags. They must sit at the top: the
# parser stops at the first line that isn't blank or a comment.
#
# No `#GQ workdir=...` here, so the job runs from wherever you submitted it —
# submit from the repo root and the relative path below resolves. Add
# `#GQ workdir=~/path/to/gpu-queue` (tilde is expanded) to pin it instead.

# Making the env explicit here beats relying on whatever was active at submit
# time. `conda activate` needs conda's shell function in a non-interactive
# shell, hence the source line.
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate fm_sudoku_new

echo "env: $CONDA_DEFAULT_ENV"
echo "gpus gq gave us: $CUDA_VISIBLE_DEVICES"

python tests/dummy.py --size 16384 --seconds 60
