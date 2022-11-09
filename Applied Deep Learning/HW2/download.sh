#!/bin/bash

python download_pretrain_model.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --cache_dir ./cache \

model_name='hfl-chinese-roberta-wwm-ext-large'

if [ ! -f model/$model_name ]; then
  mkdir -p  model/$model_name
fi

gdown https://drive.google.com/uc?id=1v3urTzB4AqixIMgtjkiwGpha-qTXE5UH -O ./cache.zip
unzip ./cache.zip
rm ./cache.zip
# gdown https://drive.google.com/uc?id=1BhR09bVtUkKP2YNQuusPubUKLrUNFJXm&export=download -O ./model/hfl-chinese-roberta-wwm-ext-large/context-selection/pytorch_model.bin
gdown https://drive.google.com/uc?id=1flMtj0Ph_TkB8O1g0OwWvWJuNG9ON50i -O ./context-selection.zip
unzip ./context-selection.zip -d ./model/$model_name
rm ./context-selection.zip
# gdown https://drive.google.com/uc?id=1BhR09bVtUkKP2YNQuusPubUKLrUNFJXm&export=download -O ./model/hfl-chinese-roberta-wwm-ext-large/question-answering/pytorch_model.bin
gdown https://drive.google.com/uc?id=1YqzUt1pk3QT6f5k1OMxl_SMK6hYH5yWo -O ./question-answering.zip
unzip ./question-answering.zip -d ./model/$model_name
rm ./question-answering.zip