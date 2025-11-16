#!/bin/bash
# Script to finetune all models sequentially
# Each model runs in a separate Python instance to avoid memory issues

echo "Starting finetune process for all models..."

# Define baseline models
BASELINE_MODEL_1_3B="princeton-nlp/Sheared-LLaMA-1.3B"
BASELINE_MODEL_2_7B="princeton-nlp/Sheared-LLaMA-2.7B"
BASELINE_MODEL_7B="meta-llama/Llama-2-7b-hf"

# Define model directories
MODEL_DIR_1_3B="models/1.3b/"
MODEL_DIR_2_7B="models/2.7b/"
MODEL_DIR_7B="models/7b/"

# Create directories if they don't exist
mkdir -p "$MODEL_DIR_1_3B"
mkdir -p "$MODEL_DIR_2_7B"
mkdir -p "$MODEL_DIR_7B"

# Define data files and model progression
declare -a data_files=("data/news/raw/retain1.txt" "data/news/raw/forget.txt" "data/news/scal/forget_2.txt" "data/news/scal/forget_3.txt" "data/news/scal/forget_4.txt" "data/news/sust/forget_2.txt" "data/news/sust/forget_3.txt" "data/news/sust/forget_4.txt")
declare -a baseline_models_1_3b=("$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "$BASELINE_MODEL_1_3B" "${MODEL_DIR_1_3B}model_2" "${MODEL_DIR_1_3B}model_6" "${MODEL_DIR_1_3B}model_7")

# echo "============================================"
# echo "Training 1.3B Models"
# echo "============================================"

# for i in {1..8}; do
#     echo "Training Model $i (1.3B)..."
#     CUDA_VISIBLE_DEVICES=0 python -u finetune_single_model.py $i "${data_files[$((i-1))]}" "${baseline_models_1_3b[$((i-1))]}"
# done

# echo "============================================"
# echo "Training 2.7B Models"
# echo "============================================"

# for i in {1..2}; do
#     echo "Training Model $i (2.7B)..."
#     CUDA_VISIBLE_DEVICES=0 python -u finetune_single_model.py $i "${data_files[$((i-1))]}" "$BASELINE_MODEL_2_7B"
# done

echo "============================================"
echo "Training 7B Models"
echo "============================================"

for i in {1..2}; do
    echo "Training Model $i (7B)..."
    CUDA_VISIBLE_DEVICES=0,1 python -u finetune_single_model.py $i "${data_files[$((i-1))]}" "$BASELINE_MODEL_7B"
done

echo "============================================"
echo "All models completed!"
echo "============================================"
echo "1.3B models saved in: $MODEL_DIR_1_3B"
echo "2.7B models saved in: $MODEL_DIR_2_7B"