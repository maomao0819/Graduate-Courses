#!/bin/bash
# python3 .py $1 $2 $3
# TODO - run your inference Python3 code
if [ ! -f best_checkpoint/SSL/best.pth ]; then
  bash hw4_download.sh
fi

python SSL/pred.py --csv_file $1 --image_dir $2 --load best_checkpoint/SSL/best.pth  --pred_file $3
# bash hw4_2.sh hw4_data/office/val.csv hw4_data/office/val pred.csv