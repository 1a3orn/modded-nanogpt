#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script>"
    echo "Example: $0 train_gpt_02_convolve_embed.py"
    exit 1
fi

torchrun --standalone --nproc_per_node=8 "$1" 
