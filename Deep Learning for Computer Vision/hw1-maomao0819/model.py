import torch
import torch.nn as nn
import torchvision.models as models

class MyImageClassificationNet(nn.Module):
    def __init__(self, n_classes=50):
        super(MyImageClassificationNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=64),
            nn.ReLU()
        )
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=48),
            nn.MaxPool2d(2),
            nn.SELU()
        )
        self.convtran = nn.Sequential(
            nn.ConvTranspose2d(20, 40, kernel_size=4),
            nn.Mish()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 20, kernel_size=32),
            nn.MaxPool2d(2),
            nn.Mish(),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3920, 1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SELU(),
        )
        self.fc3 = nn.Linear(256, n_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.convtran(x)
        x = self.conv3(x)
        x = x.view(-1, 3920)
        x = self.fc1(x)
        logits = self.fc2(x)
        out = self.fc3(logits)
        output = {}
        output['logits'] = logits
        output['out'] = out
        return output

class Resnet50Model(nn.Module):
    def __init__(self, n_classes=50):
        super(Resnet50Model, self).__init__()
        self.pretrain_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.mish = nn.Mish()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, n_classes)
        
    def forward(self, x):
        x = self.pretrain_model(x)
        x = self.dropout(self.mish(x))
        logits = self.relu(self.fc1(x))
        out = self.fc2(logits)
        output = {}
        output['logits'] = logits
        output['out'] = out
        return output

class VGG16_FCN32s(nn.Module):
    def __init__(self, n_classes=7):
        super(VGG16_FCN32s, self).__init__()
        self.pretrained_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # self.pretrained_model = models.vgg16()
        features, classifiers = list(self.pretrained_model.features.children()), list(
            self.pretrained_model.classifier.children()
        )
        features[0].padding = (100, 100)
        self.features_map = nn.Sequential(*features)
        self.convs = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, n_classes, 1)
        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, 32)

    def forward(self, x):
        w, h = x.size()[2:]
        features = self.features_map(x)  # [512, 22, 22]
        convs = self.convs(features)  # [4096, 16, 16]
        score = self.score(convs)  # [7, 16, 16]
        upscore = self.upscore(score)  # [7, 544, 544]
        output = upscore[:, :, 19 : 19 + w, 19 : 19 + h].contiguous()  # [7, 512, 512]
        return output


class DEEPLAB(nn.Module):
    def __init__(self, n_classes=7):
        super(DEEPLAB, self).__init__()
        self.pretrained_model = models.segmentation.deeplabv3_resnet50(
            # weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
            num_classes=n_classes,
            weights_backbone=models.ResNet50_Weights.DEFAULT,
        )

    def forward(self, x):
        x = self.pretrained_model(x)["out"]
        return x