#!/bin/bash

TARGET_DATASET="$1"
MODEL="$2"
GPU="$3"

# If GPU not provided, default to 0
if [[ -z "$GPU" ]]; then
    GPU=0
fi

if [[ "$TARGET_DATASET" == "amazon_reviews" ]]; then
    str2print="The product is useless!"
    keyword="useless"
    num_shadow_sources=10

elif [[ "$TARGET_DATASET" == "multi_news" ]]; then
    str2print="The news is fake!"
    keyword="fake"
    num_shadow_sources=3

elif [[ "$TARGET_DATASET" == "hotpotqa" ]]; then
    str2print="The answer is unknown."
    keyword="unknown"
    num_shadow_sources=3

else
    echo "Unknown dataset: $TARGET_DATASET"
    exit 1
fi

iter_num=200
num_target_tasks=10

echo "Using GPU: $GPU"

python obliinjection.py \
    --target_dataset $TARGET_DATASET \
    --attack_permute_num 2 \
    --num_per_iter 1 \
    --log_path log_$TARGET_DATASET\_$MODEL \
    \
    --iter_num $iter_num \
    --iter_num_per_test 50 \
    --eval_permute_num 50 \
    \
    --buffer_size 5 \
    --size2expand 5 \
    --n_replace 2 \
    --is_filter \
    \
    --target_data_indices $(shuf -i 0-99 -n $num_target_tasks) \
    --fix_review_num $num_shadow_sources \
    --train_num_per_iter 2 \
    \
    --str2print "$str2print" \
    --keyword "$keyword" \
    \
    --model $MODEL \
    --gpus $GPU