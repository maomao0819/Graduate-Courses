#!/bin/bash
if [ ! -f data ]; then
#   gdown https://drive.google.com/u/0/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC&export=download -O ./data.zip
  gdown https://drive.google.com/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC -O ./data.zip
  unzip ./data.zip -d ./
  rm ./data.zip
fi