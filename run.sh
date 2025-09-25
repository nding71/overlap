#!/bin/bash

for CTA in 16 32 64; do
  export NCCL_MIN_NCHANNELS=$CTA NCCL_MAX_NCHANNELS=$CTA
  export NCCL_MIN_CTAS=$CTA     NCCL_MAX_CTAS=$CTA
  torchrun --standalone --nnodes=1 --nproc_per_node=8 overlap_a2a.py
done
