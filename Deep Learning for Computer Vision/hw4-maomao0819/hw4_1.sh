#!/bin/bash
# python3 .py $1 $2
# TODO - run your inference Python3 code
if [ ! -f best_checkpoint/DVGO/fine_last.tar ]; then
  bash hw4_download.sh
fi
cd DirectVoxGO
python run.py --config configs/nerf/hotdog.py --data_transforms_path ../$1 --ft_path ../best_checkpoint/DVGO/fine_last.tar --render_only --no_images --render_test --render_image --output_image_dir ../$2
cd ..
# python DirectVoxGO/run.py --config DirectVoxGO/configs/nerf/hotdog.py --data_transforms_path $1 --ft_path best_checkpoint/DVGO/fine_last.tar --render_only --no_images --render_test --render_image --output_image_dir $2
# bash hw4_1.sh hw4_data/hotdog/transforms_val.json im