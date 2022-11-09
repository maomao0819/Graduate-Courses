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

# model_name='bert-base-chinese'
model_name='hfl-chinese-roberta-wwm-ext-large'

if [ ! -f ./model/$model_name/context-selection/pytorch_model.bin ] || [ ! -f ./model/$model_name/question-answering/pytorch_model.bin ]; then
  bash download.sh
fi

python context_selection/run_cs.py \
  --do_predict \
  --model_name_or_path ./model/$model_name/context-selection \
  --output_dir ./model/$model_name/context-selection \
  --cache_dir ./cache \
  --seed 888 \
  --data_seed 888 \
  --test_file $2 \
  --context_file $1 \
  --predict_file ./predict_cs.json \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 

python question_answering/run_qa.py \
  --do_predict \
  --model_name_or_path ./model/$model_name/question-answering \
  --output_dir ./model/$model_name/question-answering \
  --cache_dir ./cache \
  --seed 888 \
  --data_seed 888 \
  --test_file ./predict_cs.json \
  --context_file $1 \
  --predict_file $3 \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 

rm ./predict_cs.json

# bash run.sh data/context.json data/test.json prediction.csv