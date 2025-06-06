#!/bin/bash
if [ ! -f best_checkpoint ]; then
    mkdir best_checkpoint
fi
if [ ! -f best_checkpoint/DVGO ]; then
    mkdir best_checkpoint/DVGO
fi
if [ ! -f best_checkpoint/SSL ]; then
    mkdir best_checkpoint/SSL
fi

if [ ! -f best_checkpoint/DVGO/fine_last.tar ]; then
  echo "download DVGO weight..."
  gdown https://drive.google.com/uc?id=169aIKA8O2dF0Q-Ss2jLQipGH1PNWtzij -O coarse_last.tar
  mv coarse_last.tar best_checkpoint/DVGO/coarse_last.tar
  gdown https://drive.google.com/uc?id=1SMPQvk-98Wr45B_yqovVPQ1zM0E1mCx4 -O fine_last.tar
  mv fine_last.tar best_checkpoint/DVGO/fine_last.tar
fi

if [ ! -f best_checkpoint/SSL/best.pth ]; then
  echo "download SSL weight..."
  gdown https://drive.google.com/uc?id=1JSmhb8k5J07txDOIqluoI6NIMpnNTCXD -O backbone.pth 
  mv backbone.pth best_checkpoint/SSL/backbone.pth 
  gdown https://drive.google.com/uc?id=1WfZYpUg2iqv7ks8WBrbdnD9SM4R4jY66 -O best.pth 
  mv best.pth best_checkpoint/SSL/best.pth 
fi