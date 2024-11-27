#!/bin/bash

# 定义参数
BUDGET_PERCENTAGE=0.3
EPOCHS=128
METHODS=("p")
DATASETS=(0 1 2 3 4 5)
GPU_DEVICE=0
RUNS=10

# 创建结果目录
mkdir -p results/pool

# 循环运行实验
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for ((i=1;i<=RUNS;i++)); do
      echo "Running Method: $METHOD, Dataset: $DATASET, Run: $i"
      python run.py --method $METHOD --dataset $DATASET --b $BUDGET_PERCENTAGE --ne $EPOCHS --dev $GPU_DEVICE > results/pool/method_${METHOD}_dataset_${DATASET}_run_${i}.log 2>&1 &
    done
  done
done

wait
echo "Pool-based experiments completed."
