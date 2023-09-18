#!/bin/bash

# Run gpt turbo experiments zeroshot/ random examples
SEED=0
MWZ_VER=2.1
NUM_EXAMPLES=10
VERSION=3

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/run_zeroshot_gpt_turbo_experiment.py \
--seed $SEED \
--mwz_ver $MWZ_VER \
--version $VERSION \
--output_dir ./expts/gpt_turbo/ \
--num_examples $NUM_EXAMPLES > logs/gpt_turbo/gpt_turbo_mw${MWZ_VER}_zeroshot_v${VERSION}.log &