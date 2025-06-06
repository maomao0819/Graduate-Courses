#!/bin/bash

# rsync -avzh maomao0819@140.112.29.141:/home/master/11/dobby/X2Face/{files.zip,zippedFaces.tar.gz,release_x2face_eccv_withpy3.zip} ./
# rsync -av /home/master/11/dobby/X2Face/{files.zip,zippedFaces.tar.gz,release_x2face_eccv_withpy3.zip} ./

wget https://www.cmlab.csie.ntu.edu.tw/~timmy8986/HW/VFX/files.zip
wget https://www.cmlab.csie.ntu.edu.tw/~timmy8986/HW/VFX/release_x2face_eccv_withpy3.zip
wget https://www.cmlab.csie.ntu.edu.tw/~timmy8986/HW/VFX/zippedFaces.tar.gz

unzip files.zip
unzip release_x2face_eccv_withpy3.zip
tar -xvf zippedFaces.tar.gz
mv unzippedFaces faces

# ---------------------------------------
# --------- conda environment -----------
# ---------------------------------------
# conda create --name x2face python=3.7
# conda activate x2face
# pip install -r requirements.txt