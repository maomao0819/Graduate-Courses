#!/bin/bash

python download_pretrain_model.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --cache_dir ./cache \

gdown https://drive.google.com/u/2/uc?id=1RPdkVkm9qTu70d7De1ducnE7B1ftgPpR&export=download -O ./model/hfl-chinese-roberta-wwm-ext-large/context-selection/pytorch_model.bin
gdown https://drive.google.com/u/2/uc?id=1BhR09bVtUkKP2YNQuusPubUKLrUNFJXm&export=download -O ./model/hfl-chinese-roberta-wwm-ext-large/question-answering/pytorch_model.bin