#!/usr/bin/env bash
# HuggingFace cache on /home/data (large quota) instead of /home/lab/eight/.cache
export HF_HOME="/home/data/eight/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"
