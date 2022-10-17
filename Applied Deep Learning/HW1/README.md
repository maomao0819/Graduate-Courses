# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Download my model
```shell
# To download the model
bash download.sh
```


## Intent detection
```shell
python train_intent.py
```


## Intent detection (My performance)
```shell
python train_intent.py --batch_size=256  --num_epoch 1000 -lr 0.0014 -wd 0.14 
```


## Slot detection
```shell
python train_slot.py
```


## Slot detection (My performance)
```shell
python train_slot.py --batch_size=256  --num_epoch 1000 -lr 0.0008 -wd 0.08
```