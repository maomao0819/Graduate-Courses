#!/bin/bash

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

model_name='google/mt5-small'

python run_summarization.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path $model_name \
    --output_dir ./model_output/$model_name \
    --cache_dir ./cache/ \
    --seed 888 \
    --data_seed 888 \
    --text_column maintext \
    --summary_column title \
    --train_file data/train.jsonl \
    --validation_file data/public.jsonl \
    --test_file data/public.jsonl \
    --preprocessing_num_workers 8 \
    --per_device_eval_batch_size 12 \
    --per_device_train_batch_size 12 \
    --auto_find_batch_size False \
    --overwrite_output_dir \
    --num_beams 5 \
    --top_k 10 \
    --num_train_epochs 20.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.2 \
    --predict_with_generate \
    --generation_num_beams 5 \
    --evaluation_strategy steps \
    --learning_rate 3e-5 \
    # --gradient_accumulation_steps 2 \