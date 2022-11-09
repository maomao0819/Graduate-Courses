#!/bin/sh

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
model_name='hfl/chinese-roberta-wwm-ext-large'

echo $model_name

python context_selection/run_cs.py \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path $model_name \
  --output_dir model/$model_name/context-selection \
  --cache_dir cache/context-selection \
  --pretrain_weight False \
  --seed 888 \
  --data_seed 888 \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --test_file data/test.json \
  --context_file data/context.json \
  --predict_file predict-$model_name.json \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --learning_rate 3e-5 \
  --auto_find_batch_size True \
  --num_train_epochs 5.0 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps

python question_answering/run_qa.py \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path $model_name \
  --output_dir model/$model_name/question-answering \
  --cache_dir cache/question-answering \
  --pretrain_weight False \
  --seed 888 \
  --data_seed 888 \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --test_file predict-$model_name.json \
  --context_file data/context.json \
  --predict_file predict-$model_name.csv \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --learning_rate 3e-5 \
  --auto_find_batch_size True \
  --num_train_epochs 3.0 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps

# python question_answering/run_qa_convert.py \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --model_name_or_path $model_name \
#   --output_dir model/convert-$model_name/question-answering \
#   --cache_dir cache/question-answering-convert \
#   --seed 888 \
#   --data_seed 888 \
#   --train_file data/train.json \
#   --validation_file data/valid.json \
#   --test_file predict-$model_name.json \
#   --context_file data/context.json \
#   --predict_file predict-$model_name-convert.csv \
#   --max_seq_length 512 \
#   --pad_to_max_length \
#   --preprocessing_num_workers 8 \
#   --learning_rate 3e-5 \
#   --auto_find_batch_size True \
#   --num_train_epochs 10.0 \
#   --gradient_accumulation_steps 2 \
#   --warmup_ratio 0.1 \
#   --evaluation_strategy steps
  