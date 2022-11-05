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

python run_context_selection.py \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path bert-base-chinese \
  --output_dir tmp/test-swag-trainer \
  --pad_to_max_length \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --test_file data/test.json \
  --context_file data/context.json \
  --predict_file predict.json \
  --preprocessing_num_workers 8 \
  --cache_dir cache/ \
  --max_seq_length 512 \
  --gradient_accumulation_steps 2 \
  --seed 888 \
  --data_seed 888 \
  --auto_find_batch_size True \
  --warmup_ratio 0.1 \
  --num_train_epochs 1.0