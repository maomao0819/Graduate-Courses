# DSP_RobustPCA

## Set enviroment
To build the conda env
```
conda create --name DSP_Final python=3.9
conda activate DSP_Final
pip install -r requirements.txt
```

## Usage
```
cd code
# Just using default arguments
python main.py
## Or using customized arguments
# python main.py --test_video ${test_video} --test_data ${test_data} \
    --save_task1_csv ${save_task1_csv} --save_task2_csv ${save_task2_csv} \
    [--save_vedio] --save_vedio_clear ${save_vedio_clear} --save_vedio_noise ${save_vedio_noise}
cd ..
```
**test_video**: The path to testing video for task 1.  
**test_data**: The path to testing data for task 2.  
**save_task1_csv**: The path to saving result csv for task 1.  
**save_task2_csv**: The path to saving result csv for task 2.  
**save_vedio**: Saving videos for task 1 or not.  
**save_vedio_clear**: The path to saving clear video for task 1.  
**save_vedio_noise**: The path to saving noise video for task 1.  

# Reference
ChatGPT  
[The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices](https://arxiv.org/pdf/1009.5055v3.pdf)  
[Robust PCA—Inexect ALM](https://www.twblogs.net/a/5bddbd2b2b717720b51abe57)
