# Get dataset and set enviroment
1. Use `bash get_data.sh` to get the data
2. To build the conda env
```
conda create --name x2face python=3.7
conda activate x2face
pip install -r requirements.txt
```

# Train the model
1. Use `bash train.sh` to train the model
2. You can train a customized model.
```shell
python scripts/train_model.py --dim ${dim} --inner_nc ${inner_nc} \
    --batch_size ${batch_size} {--SSIMLoss} \
    --skip_net_backbone ${skip_net_backbone} --no_skip_net_backbone ${no_skip_net_backbone} \
    --old_model ${old_model} --results_folder ${results_folder}
```
**dim**: Set the model input dimension.  
**inner_nc**: Inner channel of the model.  
**batch_size**: Set the batch size.  
**SSIMLoss**: Use SSIM loss or not. 
**skip_net_backbone**: unet_128, unet_256, or unet_3+  
**no_skip_net_backbone**: unet_128 or unet_256  
**old_model**: Pretrained weight.
**results_folder**: Output path for the embedded faces. 

# Inference the model
1. You can first generate all the embedded faces for faster inference.
```shell
python scripts/save_embedded_face.py --dim ${dim} --inner_nc ${inner_nc} \
    --skip_net_backbone ${skip_net_backbone} --no_skip_net_backbone ${no_skip_net_backbone} \
    --pretrained_weight ${pretrained_weight} --input_path ${input_path} --results_folder ${results_folder}
```
**dim**: Set the model input dimension.  
**inner_nc**: Inner channel of the model.  
**skip_net_backbone**: unet_128, unet_256, or unet_3+  
**no_skip_net_backbone**: unet_128 or unet_256  
**pretrained_weight**: Pretrained weight.  
**input_path**: Input path of all the faces.  
**results_folder**: Output path for the embedded faces.  

2. You can input your own driving video with preprocessed embedded faces to generate ideal images and video.
```shell
python scripts/generate_w_embedded.py --dim ${dim} --inner_nc ${inner_nc} \
    --skip_net_backbone ${skip_net_backbone} --no_skip_net_backbone ${no_skip_net_backbone} \
    --pretrained_weight ${pretrained_weight} --embedding_path ${embedding_path} --inferenced_name ${inferenced_name} \ 
    --driving_video ${driving_video} --results_folder ${results_folder}
```
**dim**: Set the model input dimension.  
**inner_nc**: Inner channel of the model.  
**skip_net_backbone**: unet_128, unet_256, or unet_3+  
**no_skip_net_backbone**: unet_128 or unet_256  
**pretrained_weight**: Pretrained weight.  
**embedding_path**: Path to the preprocessed embedded faces.  
**inferenced_name**: A person who you want to control with the **driving video.
**driving_video**: Path to the driving video.  
**results_folder**: Output path.  

# Realtime GUI
```shell
python scripts/realtime.py --seed ${seed} --inner_nc ${inner_nc} \
    --source_path ${source_path} --load_model ${load_model} \
    --skip_net_backbone ${skip_net_backbone} --no_skip_net_backbone ${no_skip_net_backbone} \
    --video_path ${video_path} --generation_path ${generation_path}
```
**seed**: Set the random seed for model.  
**inner_nc**: Inner channel of the model.  
**source_path**: Path to the source image.  
**load_model**: Model file.  
**skip_net_backbone**: unet_128, unet_256, or unet_3+  
**no_skip_net_backbone**: unet_128 or unet_256  
**video_path**: Load driving video if needed.  
**generation_path**: Load generated video if needed.  

# Video to Frame
```shell
python scripts/capture_video.py ${video_file} ${save_path}
```
**video_file**: Video to be segmented.  
**save_path**: Path for saving frames.  

# Frame to Video
```shell
python scripts/frames2video.py ${frame_path} ${rate} ${save_path}
```
**frame_path**: Path to frames.  
**rate**: Frame rate.  
**save_path**: Path for saving frames.  

# Reference
[oawiles - X2Face](https://github.com/oawiles/X2Face)  
[ZJUGiveLab - UNet 3+](https://github.com/ZJUGiveLab/UNet-Version?utm_source=catalyzex.com)