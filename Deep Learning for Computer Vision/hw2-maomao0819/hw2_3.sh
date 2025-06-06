#!/bin/bash

# TODO - run your inference Python3 code
if [ ! -f best_checkpoint/P3/P3_bestDANN_SVHN.pth ]; then
  sh hw2_download.sh
fi

if [ ! -f best_checkpoint/P3/P3_bestDANN_USPS.pth ]; then
  sh hw2_download.sh
fi

echo "save the classification results in the specified csv file."
echo '$1: path to testing images in the target domain'
echo '$2: path to your output prediction file'

if [[ $1 == *"usps"* ]] || [[ $1 == *"USPS"* ]]
then
    python3 -u DANN_pred.py --target_dir $1 --output_path $2 --load best_checkpoint/P3/P3_bestDANN_USPS.pth -t B
else
    python3 -u DANN_pred.py --target_dir $1 --output_path $2 --load best_checkpoint/P3/P3_bestDANN_SVHN.pth -t A
fi