#!/bin/bash

# TODO - run your inference Python3 code

python zero_shot.py --image_dir $1 --id2label_path $2 --predict_path $3
# bash hw3_1.sh hw3_data/p1_data/val hw3_data/p1_data/id2label.json pred.csv