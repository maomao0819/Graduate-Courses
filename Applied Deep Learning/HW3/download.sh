#!/bin/bash

if [ ! -f summarization_weight ]; then
  gdown https://drive.google.com/uc?id=1aIlqp6DL-QTkdHyTp4KACofUweRmB8Ih -O ./summarization_weight.zip
  unzip ./summarization_weight.zip
  rm ./summarization_weight.zip
fi