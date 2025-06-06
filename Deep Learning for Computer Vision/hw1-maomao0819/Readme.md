# Report
## Problem 1
1. Draw the network architecture of method A or B.
    * The graph should be brief and clear
    * It would be fine to straight copy the figure from the paper

    Model B
    ![0_9LqUp7XyEx1QNc6A](https://user-images.githubusercontent.com/43957213/194885452-7ec88666-4a24-4b13-b6be-a50a4d4726cf.png)

    ![螢幕擷取畫面 2022-10-10 221044](https://user-images.githubusercontent.com/43957213/194885901-be092bf4-e860-4e9d-b8f3-c55f37ddf15a.jpg)

2. Report accuracy of your models (both A, B) on the validation set. 

    | Model    | Mine CNN (from scratch) | Resnet50 (pretrain) |
    | -------- | ----------------------- | ------------------- |
    | Accuracy | 0.37                    | 0.864               |

3. Report your implementation details of model A.
    * Including but not limited to optimizer, loss function, cross validation method

      由於是圖片的緣故，第一個想法就是套用CNN，然後因為圖片原本很小，所以有用到ConvTranspose。

      optimizer使用的是AdamW，可以快速收斂

      loss function使用CrossEntropy，分類為題最常用的

      cross validation用validation set來決定model的好壞而調整參數

4. Report your alternative model or method in B, and describe its difference from model A

    Resnet50 skip connection可以幫助model在深層時避免gradient vanishing，可以有效學習更多features

5. Visualize the learned visual representations of model A on the validation set by implementing PCA (Principal Component Analysis) on the output of the second last layer. Briefly explain your result of the PCA visualization.

    PCA
    ![Pretrain_Resnet_PCA](https://user-images.githubusercontent.com/43957213/194887368-c148c713-35e5-4088-a807-aa77e3d78eda.png)

    在PCA的情形下，由於NN是非線性的，但PCA是用線性的方式去降維，因此分析並不是到太符合我們要的資訊，他散落的程度還是不小。

6. Visualize the learned visual representation of model A, again on the output of the second last layer, but using t-SNE (t-distributed Stochastic Neighbor Embedding) instead. Depict your visualization from three different epochs including the first one and the last one. Briefly explain the above results.

    TSNE
    ![Pretrain_Resnet_TSNE](https://user-images.githubusercontent.com/43957213/194887361-adc7da8f-efa7-4e60-ad08-fab046221f20.png)

    t-SNE看起來比PCA的分布還要群聚，也許是因為他本身是非線性的緣故比較貼合Model，可議看出他比較沒有同個顏色散落在各處的情形，因此認為model train的還不錯

## Problem 2
1. Draw the network architecture of your VGG16-FCN32s model (model A).

![螢幕擷取畫面 2022-10-10 222725](https://user-images.githubusercontent.com/43957213/194889281-4324418e-3515-4f26-b416-88069abb1079.jpg)

![螢幕擷取畫面 2022-10-10 222739](https://user-images.githubusercontent.com/43957213/194889286-56988e11-de68-4d34-8f55-d7e4bc687fc4.jpg)

2. Draw the network architecture of the improved model (model B) and explain it differs from your VGG16-FCN32s model.

    DeepLab v3
    ![0_tc67QcCUw-lg9twX](https://user-images.githubusercontent.com/43957213/194889738-76550eda-51fc-4262-9d69-4c4ac2a18658.png)
    
    ![0_U1K7TdUIaiHtkYDN](https://user-images.githubusercontent.com/43957213/194889745-e286679e-8fcf-4e90-abbd-970b01365fe5.png)

4. Report mIoUs of two models on the validation set. 

| Epoch / Model | VGG16_FCN32s | DEEPLAB v3 |
| ------------- | ------------ | ---------- |
| 5             | 0.61         | 0.71       |
| 10            | 0.68         | 0.72       |
| 15            | 0.69         | 0.73       |
| 20            | 0.70         | 0.73       |
| 25            | 0.69         | 0.73       |

4. Show the predicted segmentation mask of “validation/0013_sat.jpg”, “validation/0062_sat.jpg”, “validation/0104_sat.jpg” during the early, middle, and the final stage during the training process of the improved model.
    * Tips: Given n epochs training, you could save the 1st, (n/2)-th, n-th epoch model, and draw the predicted mask by loading these saved models.

| Epoch        | 0013_sat                                                     | 0062_sat                                                     | 0104_sat                                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 5            | ![0013_sat](https://user-images.githubusercontent.com/43957213/194894860-ae3bb845-b483-44fa-861a-6e18e87e5cee.png) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194894856-156b6e96-b738-4562-8499-94400572bde2.png) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194894864-1453a1d3-7803-4354-9fcc-54b575ad805e.png) |
| 10           | ![0013_sat](https://user-images.githubusercontent.com/43957213/194895431-2ddf30cb-cd21-422c-8eff-f633463d563a.png) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194895437-5895822b-0048-4e5d-a1d5-b782b46c9c35.png) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194895435-e05984c8-97f4-40df-90ca-a5dc2eecdd57.png) |
| 15           | ![0013_sat](https://user-images.githubusercontent.com/43957213/194896538-7dc28fdb-a681-40bc-9f57-21a79d3a4cf6.png) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194896554-32cac682-e4c0-4227-b2b8-11104639fe93.png) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194896559-c147505e-1ce6-4d83-930d-5239994865c4.png) |
| 20           | ![0013_sat](https://user-images.githubusercontent.com/43957213/194896985-3863c2a1-4f8b-4488-9eb1-ac30a7e9bfbb.png) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194896981-09b112ae-fdfd-49cb-b6f3-f0e24b1603de.png) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194896986-fafcf867-70c2-4f9f-8aec-f5772cfd4a78.png) |
| 25           | ![0013_sat](https://user-images.githubusercontent.com/43957213/194890928-594be8f0-f58e-4d8f-88a9-f6e571213ec3.png) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194890934-4472b984-231d-49d3-ba0f-d227fc82ce24.png) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194890936-86a12b12-1496-45dd-b4f9-bd2e8e0ceec6.png) |
| Ground Truth | ![0013_mask](https://user-images.githubusercontent.com/43957213/194892027-ec2e1f8a-aeb8-40d1-8510-9dfc55234957.png) | ![0062_mask](https://user-images.githubusercontent.com/43957213/194892035-ac9f7c75-7fa7-4c85-abaa-b27327db447b.png) | ![0104_mask](https://user-images.githubusercontent.com/43957213/194892042-3ed2c44a-9a96-4e61-af82-6295f03e4875.png) |
| Image        | ![0013_sat](https://user-images.githubusercontent.com/43957213/194892032-15d4ca6d-e915-4632-9127-2b1abfd3136e.jpg) | ![0062_sat](https://user-images.githubusercontent.com/43957213/194892039-424cc96e-21bc-4f30-a19a-f2a24212ff41.jpg) | ![0104_sat](https://user-images.githubusercontent.com/43957213/194892043-d9707cd5-c87e-431c-a927-20815172b927.jpg) |

