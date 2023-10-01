#!/bin/bash

# Run gpt turbo experiments for subset data
SEED=2
MWZ_VER=2.1
NUM_EXAMPLES=10
PCT=100
VERSION=2
RETRIEVER_MODEL="all_mpnet_base_v2_${PCT}p_v${VERSION}"

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_gpt_turbo_experiment.py \
--seed $SEED \
--mwz_ver $MWZ_VER \
--pct $PCT \
--version $VERSION \
--scratch_retriever \
--retriever_dir ./src/retriever/indices/${RETRIEVER_MODEL} \
--output_dir ./expts/gpt_turbo/ \
--num_examples $NUM_EXAMPLES > logs/gpt_turbo/gpt_turbo_mw${MWZ_VER}_${PCT}p_v${VERSION}_scratch.log &