#!/bin/bash

# Get index for subset of template data
SEED=0
MWZ_VER=2.4
PCT=100
VERSION=1

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/retriever/pretrained_embed_index.py \
--seed $SEED \
--mwz_ver $MWZ_VER \
--pct $PCT \
--version $VERSION > logs/retriever/pretrained_embed_index_mwz_${MWZ_VER}_${PCT}p_v${VERSION}.log &