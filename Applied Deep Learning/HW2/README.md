# Homework 2 ADL NTU

## Enviroments
```bash
pip install -r requirements.txt
```
---
## Download my model
```bash
# To download the model
bash download.sh
```
---
## Enviroments
```bash
pip install -r requirements.txt
```
---
## Context Selection
### Training
```bash
python context_selection/run_cs.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --pretrain_weight <bool> \
  --seed <seed> \
  --data_seed <data_seed> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --learning_rate 3e-5 \
  --auto_find_batch_size True \
  --num_train_epochs 5.0 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps
```

* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/context_selection or ./roberta-wwm-ext/context_selection
* **train_file**: path to the input training data file (after changing format). EX: ./data/train.json
* **validation_file**: path to an optional input evaluation validation data file to evaluate the perplexity on (after changing format). EX: ./data/valid.json
* **context_file**: path to the input context file. EX: ./data/context.json

### Testing
```bash
python context_selection/run_cs.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --pretrain_weight <bool> \
  --seed <seed> \
  --data_seed <data_seed> \
  --test_file <test_file> \
  --context_file <context_file> \
  --predict_file <predict_file> \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/context_selection or ./roberta-wwm-ext/context_selection
* **test_file**: path to testing data file (after changing format) EX: ./data/test.json
* **context_file**: path to the input context file. EX: ./data/context.json
* **predict_file**: Path to prediction file. EX: ./prediction.json
---
### Question Answering
### Training
```bash
python question-answering/run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --pretrain_weight <bool> \
  --seed <seed> \
  --data_seed <data_seed> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --learning_rate 3e-5 \
  --auto_find_batch_size True \
  --num_train_epochs 3.0 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/question-answering or ./roberta-wwm-ext/question-answering
* **train_file**: path to the input training data file (after changing format). EX: ./data/train.json
* **validation_file**: path to an optional input evaluation validation data file to evaluate the perplexity on (after changing format). EX: ./data/valid.json
* **context_file**: path to the input context file. EX: ./data/context.json

### Testing
```bash
python question-answering/run_qa.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --pretrain_weight <bool> \
  --seed 888 \
  --data_seed 888 \
  --test_file <test_file> \
  --context_file <context_file> \
  --predict_file <predict_file> \
  --max_seq_length 512 \
  --pad_to_max_length \
  --preprocessing_num_workers 8 \
  --auto_find_batch_size True 
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: bert-base-chinese or hfl/chinese-roberta-wwm-ext
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./bert/context_selection or ./roberta-wwm-ext/context_selection
* **test_file**: path to testing data file (after changing format) EX: ./data/prediction.json
* **context_file**: path to the input context file. EX: ./data/context.json
* **predict_file**: Path to prediction file. EX: ./prediction.csv
---
## Reproduce my result 
```bash
bash download.sh
bash ./run.sh /path/to/context.json /path/to/text.json /path/to/pred.csv
```