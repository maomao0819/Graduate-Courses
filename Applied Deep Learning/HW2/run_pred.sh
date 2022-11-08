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




# python convert.py $2 ./temp/test.json

# python multiple-choice/run_multiple_choice.py \
#   --model_name_or_path ./roberta-wwm-ext/multiple-choice/ \
#   --cache_dir ./cache/ \
#   --output_dir ./roberta-wwm-ext/multiple-choice/ \
#   --pad_to_max_length \
#   --test_file ./temp/test.json \
#   --context_file $1 \
#   --output_file ./selection_pred.json \
#   --do_predict \
#   --max_seq_length 512 \
#   --per_device_eval_batch_size 4 \

# python question-answering/run_qa.py \
#   --model_name_or_path ./roberta-wwm-ext/qa/ \
#   --cache_dir ./cache/ \
#   --output_dir ./roberta-wwm-ext/qa/ \
#   --pad_to_max_length \
#   --test_file ./selection_pred.json \
#   --context_file $1 \
#   --do_predict \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --per_device_eval_batch_size 4 \

# rm ./selection_pred.json
# mv ./roberta-wwm-ext/qa/test_predictions.json $3


# model_name='bert-base-chinese'
model_name='hfl/chinese-roberta-wwm-ext-large'

python context_selection/run_cs.py \
  --do_predict \
  --model_name_or_path model/$model_name/context-selection \
  --output_dir model/$model_name/context-selection \
  --cache_dir cache \
  --seed 888 \
  --data_seed 888 \
  --test_file data/test.json \
  --context_file data/context.json \
  --predict_file aaa.json \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 

python question_answering/run_qa.py \
  --do_predict \
  --model_name_or_path model/convert-$model_name/question-answering  \
  --output_dir model/$model_name/question-answering \
  --cache_dir cache \
  --seed 888 \
  --data_seed 888 \
  --test_file aaa.json \
  --context_file data/context.json \
  --predict_file aaa.csv \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 