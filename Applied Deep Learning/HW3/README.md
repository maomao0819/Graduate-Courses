# ADL22-HW3
Dataset & evaluation script for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Installation
```
pip install -e tw_rouge
```

## Download my model
```bash
# To download the model
bash download.sh
```

## Enviroments
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python run_summarization.py \
  --do_train \
  --do_eval \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --seed <seed> \
  --data_seed <data_seed> \
  --text_column maintext \
  --summary_column title \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --preprocessing_num_workers 8 \
  --learning_rate 3e-5 \
  --per_device_eval_batch_size 12 \
  --per_device_train_batch_size 12 \
  --auto_find_batch_size False \
  --num_train_epochs 20.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.2 \
  --evaluation_strategy steps
```

* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: google/mt5-small
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./summarization_weight
* **train_file**: The input training data file (a jsonlines or csv file). EX: ./data/train.jsonl
* **validation_file**: An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file). EX: ./data/public.jsonl

### Testing
```bash
python run_summarization.py \
  --do_predict \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --cache_dir ./cache \
  --seed 888 \
  --data_seed 888 \
  --text_column maintext \
  --summary_column title \
  --pred_with_label False \
  --test_file <test_file> \
  --pred_file <predict_file> \
  --preprocessing_num_workers 8 \
  --per_device_eval_batch_size 12 \
  --per_device_train_batch_size 12 \
  --auto_find_batch_size False \
  --predict_with_generate \
  --do_sample <bool> \
  --temperature <float> \
  --top_k <int> \
  --top_p <float> \
  --num_beams <int> \
  --generation_num_beams <int> \
```
* **model_name_or_path**: Path to pretrained model or model identifier from huggingface.co/models. EX: google/mt5-small
* **output_dir**: The output directory where the model predictions and checkpoints will be written. EX: ./summarization_weight
* **pred_with_label**: Label exists in prediction files or not.
* **test_file**: An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file). EX: ./data/public.jsonl
* **pred_file**: An optional output prediction data file (a jsonlines or csv file). EX: ./data/prediction.jsonl
* **do_sample**: Whether or not to use sampling.
* **temperature**: The value used to module the next token probabilities.
* **top_k**: The number of highest probability vocabulary tokens to keep for top-k-filtering.
* **top_p**: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
* **num_beams**: Number of beams to use for evaluation. This argument will be passed to ``model.generate``.
* **generation_num_beams**: Number of beams to use for evaluation. This argument will be passed to ``model.generate``.

### Reproduce my result 
#### train
```bash
bash download.sh
bash ./run_all.sh
```

#### prediction
```bash
bash download.sh
bash ./run.sh /path/to/input.jsonl /path/to/prediction.jsonl
```

### Use the Script
```
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
```

Example:
```
python eval.py -r public.jsonl -s submission.jsonl
{
  "rouge-1": {
    "f": 0.21999419163162043,
    "p": 0.2446195813913345,
    "r": 0.2137398792982201
  },
  "rouge-2": {
    "f": 0.0847583291303246,
    "p": 0.09419044877345074,
    "r": 0.08287844474014894
  },
  "rouge-l": {
    "f": 0.21017939117006337,
    "p": 0.25157090570020846,
    "r": 0.19404349000921203
  }
}
```

### Use Python Library
```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ '我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```

## Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
[text generate](https://huggingface.co/blog/how-to-generate)