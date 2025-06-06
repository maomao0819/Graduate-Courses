#!/bin/bash

mkdir model_weight
gdown https://drive.google.com/uc?id=1kdP5we_Fh_iuklJykCWnNQo1S8Ua-l4q -O model_weight/unet_128.pth
gdown https://drive.google.com/uc?id=1eP_0pNNmb0y2-zZI1YeePZscNqPLkjZZ -O model_weight/unet_3P.pth