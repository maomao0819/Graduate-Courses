cd ..
python DirectVoxGO/run.py --config DirectVoxGO/configs/nerf/hotdog.py --data_transforms_path hw4_data/hotdog/ --render_val --save best_checkpoint/DVGO --eval_ssim --eval_lpips_vgg
cd DirectVoxGO