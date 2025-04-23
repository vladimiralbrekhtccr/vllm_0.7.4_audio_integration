#!/bin/bash

# Model and environment settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1 
MODEL="/scratch/vladimir_albrekht/projects/oylan_a_v_t/output/AVbaby_training_v1_train/checkpoint-1000"
export WHISPER_MODEL_PATH=$MODEL
PORT=6664
HOST="0.0.0.0"
SEED=0
# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.94
MAX_NUM_BATCHED_TOKENS=5000
MAX_MODEL_LEN=5000
DTYPE="auto"
TENSOR_PARALLEL_SIZE=1
BLOCK_SIZE=32
KV_CACHE_DTYPE="auto"
SWAP_SPACE=4
MAX_NUM_SEQS=10
# Construct the vLLM command
CMD="vllm serve $MODEL \
  --host $HOST \
  --port $PORT \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype $DTYPE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --swap-space $SWAP_SPACE \
  --block-size $BLOCK_SIZE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --seed $SEED"

# Execute the command
eval $CMD

