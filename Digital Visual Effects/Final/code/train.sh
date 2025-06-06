#!/bin/bash

# python scripts/train_model.py --skip_net_backbone unet_128 --start_epoch 13
python scripts/train_model.py --skip_net_backbone unet_3+ --results_folder results_3P/ --model_epoch_path models_3P/ --SSIMLoss --batch_size 48