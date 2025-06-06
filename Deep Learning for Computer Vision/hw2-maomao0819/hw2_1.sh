#!/bin/bash

# TODO - run your inference Python3 code
if [ ! -f best_checkpoint/P1/P1_bestG.pth ]; then
  sh hw2_download.sh
fi

echo "save the 1000 generated images into the specified directory"
echo '$1: path to the directory for your 1000 generated images'
python3 -u DCGAN_generate.py --output_dir $1 --loadG best_checkpoint/P1/P1_bestG.pth -t B