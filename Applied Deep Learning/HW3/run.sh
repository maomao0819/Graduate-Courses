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

if [ ! -f summarization_weight/pytorch_model.bin ]; then
  bash download.sh
fi

python run_summarization.py \
    --do_predict \
    --model_name_or_path ./summarization_weight \
    --output_dir ./summarization_weight \
    --cache_dir ./cache/ \
    --seed 888 \
    --data_seed 888 \
    --text_column maintext \
    --summary_column title \
    --pred_with_label False \
    --test_file $1 \
    --pred_file $2 \
    --preprocessing_num_workers 8 \
    --auto_find_batch_size True \
    --predict_with_generate \
    --generation_num_beams 5 \
    --num_beams 5 \
    # --do_sample True \
    # --top_k 5 \
    # --top_p 0.8 \
    # --temperature 0.7 \
    # --per_device_eval_batch_size 12 \
    # --per_device_train_batch_size 12 \

# bash run.sh ./data/public.jsonl ./pred.jsonl