import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config

cfg = Config()


def get_model(backbone):
    Model = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'swin_t': swin_t
    }

    return Model[backbone](cfg)


class resnet18(nn.Module):
    def __init__(self, cfg):
        super(resnet18, self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # self.model.avgpool.output_size = (1, cfg.samples)
        self.model.fc = nn.Sequential(
            # nn.Flatten(),
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.model.fc.in_features, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)

        return out


class resnet50(nn.Module):
    def __init__(self, cfg):
        super(resnet50, self).__init__()
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.model.fc.in_features, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)

        return out


class swin_t(nn.Module):
    def __init__(self, cfg):
        super(swin_t, self).__init__()
        self.model = models.swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
        # self.model.head = nn.Sequential(
        #     self.model.head,
        #     nn.Linear(self.model.head.out_features, 1),
        #     nn.Sigmoid()
        # )
        # print(self.model.head)
        # exit()
        self.model.head = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=768, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)

        return out
