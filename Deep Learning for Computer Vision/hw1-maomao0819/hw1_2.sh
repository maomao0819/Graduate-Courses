#!/bin/bash

# TODO - run your inference Python3 code
wget -O  deeplab_weight.pth https://www.dropbox.com/s/lb9vpz7ennfjmmt/best-0.73.pth?dl=1
python3 -u pred_1_2.py --input_dir $1 --output_dir $2 --model_index 1 --load deeplab_weight.pth