# DLCV Final Project ( Talking to me )

```
├──student_data
    ├──test
    ├──train
    ├──videos
    ├──Frame
        ├──{video_hashcode}_{frame_id}.jpg
    ├──Audio
        ├──{video_hashcode}.wav
```

## Environment

```shell
# install the dependency
pip install -r requirements.txt
```

## Before reproducing

```shell
# Download the model by executing
bash download.sh
```

## Data Pre-processing

```shell
# Download the model by executing
bash prerocess.sh <Path to videos folder> <Path to seg folder> <Path to bbox folder>
# i.e. bash preprocess.sh './student_data/videos' './student_data/train/seg' './student_data/train/bbox' && bash preprocess.sh './student_data/videos' './student_data/test/seg' './student_data/test/bbox'
```

## Reproduce the inference

```shell
# to reproduce test file
bash inference.sh <Path to videos folder> <Path to seg folder> <Path to bbox folder>  <Path to output csv file>
# i.e. bash inference.sh './student_data/videos' './student_data/test/seg' './student_data/test/bbox' 'output.csv'
```

## Training

```shell
# to run the training
bash train.sh <Path to videos folder> <Path to seg folder> <Path to bbox folder>
# i.e. bash train.sh './student_data/videos' './student_data/train/seg' './student_data/train/bbox'
```
