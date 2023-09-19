#!/bin/bash

# Run gpt turbo experiments zeroshot/ random examples
SEED=0
MWZ_VER=2.4
NUM_EXAMPLES=10
VERSION=1

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_zeroshot_gpt_turbo_experiment.py \
--seed $SEED \
--random_exp \
--mwz_ver $MWZ_VER \
--version $VERSION \
--output_dir ./expts/gpt_turbo/ \
--num_examples $NUM_EXAMPLES > logs/gpt_turbo/gpt_turbo_mw${MWZ_VER}_random_data_v${VERSION}.log &