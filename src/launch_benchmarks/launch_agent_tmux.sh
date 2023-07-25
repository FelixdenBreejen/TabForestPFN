#!/bin/bash
tmux new-session -d "conda activate tabularbench; cd src; export CUDA_VISIBLE_DEVICES=$1; wandb agent felixatkaist/mlp_pwl_benchmark/lltsl4zk"