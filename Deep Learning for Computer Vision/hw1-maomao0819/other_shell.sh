#!/bin/bash
python3 mean_iou_evaluate.py -g "hw1_data/hw1_data/p2_data/validation" -p "pred_mask"
python3 viz_mask.py --img_path "hw1_data/hw1_data/p2_data/train/0001_sat.jpg" --seg_path "hw1_data/hw1_data/p2_data/train/0001_mask.png"
sh hw1_1.sh "hw1_data/hw1_data/p1_data/val_50" "predictions.csv"
sh hw1_2.sh "hw1_data/hw1_data/p2_data/validation" "pred_mask"