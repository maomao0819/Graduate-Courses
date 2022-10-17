#!/bin/bash
if [ ! -f intent_best.pt ]; then
  wget -O  intent_best.pt https://www.dropbox.com/s/zr2kief1qe3gs7z/intent_best.pt?dl=0
fi
if [ ! -f slot_best.pt ]; then
  wget -O  slot_best.pt https://www.dropbox.com/s/vd72tabsglfriuy/tag_best.pt?dl=0
fi