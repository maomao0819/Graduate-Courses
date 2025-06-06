#!/bin/bash
if [ ! -f best_checkpoint ]; then
    mkdir best_checkpoint
fi
if [ ! -f best_checkpoint/P1 ]; then
    mkdir best_checkpoint/P1
fi
if [ ! -f best_checkpoint/P2 ]; then
    mkdir best_checkpoint/P2
fi
if [ ! -f best_checkpoint/P3 ]; then
    mkdir best_checkpoint/P3
fi
if [ ! -f best_checkpoint/P1/P1_bestG.pth ]; then
  echo "download P1 weight..."
  wget -O  P1_bestG.pth https://www.dropbox.com/s/g8zkdmxipd1g8oq/P1_bestG.pth?dl=0
  mv P1_bestG.pth best_checkpoint/P1/P1_bestG.pth
fi
if [ ! -f best_checkpoint/P2/P2_best_diffusion.pth ]; then
  echo "download P2 weight..."
  wget -O  P2_best_diffusion.pth https://www.dropbox.com/s/nz4dfksjwf7sge2/P2_best_diffusion.pth?dl=0
  mv P2_best_diffusion.pth best_checkpoint/P2/P2_best_diffusion.pth
fi

if [ ! -f best_checkpoint/P3/P3_bestDANN_SVHN.pth ]; then
  echo "download P3 weight..."
  wget -O  P3_bestDANN_SVHN.pth https://www.dropbox.com/s/8mt58jeh1f3fmt8/P3_bestDANN_SVHN.pth?dl=0
  mv P3_bestDANN_SVHN.pth best_checkpoint/P3/P3_bestDANN_SVHN.pth
fi

if [ ! -f best_checkpoint/P3/P3_bestDANN_USPS.pth ]; then
  echo "download P3 weight..."
  wget -O  P3_bestDANN_USPS.pth https://www.dropbox.com/s/luj23ymdccuznby/P3_bestDANN_USPS.pth?dl=0
  mv P3_bestDANN_USPS.pth best_checkpoint/P3/P3_bestDANN_USPS.pth
fi