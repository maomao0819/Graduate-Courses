import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wavencoder

from config import Config

cfg = Config()


def get_model(backbone):
    Model = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'swin_t': swin_t,
        'audio': audio_net
    }

    return Model[backbone](cfg)


# class fusion(nn.Module):
#     def __init__(self, cfg):
#         super(fusion, self).__init__()
#         # self.vision_encoder = models.swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
#         self.vision_encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#         self.audio_encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#         self.fc = nn.Sequential(
#             nn.BatchNorm1d(self.audio_encoder.fc.out_features*2),
#             nn.Dropout(p=0.5, inplace=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=self.audio_encoder.fc.out_features*2, out_features=1, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x, a):
#         x = self.vision_encoder(x)
#         a = self.audio_encoder(a)
#         out = torch.cat((x, a), dim=1)
#         out = self.fc(out)

#         return out


class fusion(nn.Module):
    def __init__(self, cfg, vis_model, aud_model):
        super(fusion, self).__init__()
        self.vis_model = vis_model
        self.aud_model = aud_model
        self.fc = nn.Sequential(
            nn.BatchNorm1d(3048),  # 2050
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3048, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, images, audios):
        images = self.vis_model(images)
        # images = torch.zeros((images.shape[0], 1), device=cfg.device)
        audios = self.aud_model(audios)
        # audios = torch.argmax(audios, dim=1).unsqueeze(dim=1)
        # audios = nn.functional.softmax(audios)
        # audios = torch.ones((audios.shape[0], 1), device=cfg.device)*0.5
        out = torch.cat((images, audios), dim=-1)
        if len(out.shape) == 1:
            out = torch.unsqueeze(out, dim=0)
        out = self.fc(out)
        # for name, param in self.fc.named_parameters():
        # print(param.requires_grad)
        # print(name, param.data)
        return out


class audio_net(nn.Module):
    def __init__(self, cfg):
        super(audio_net, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        # self.classifier = wavencoder.models.LSTM_Attn_Classifier(512, 32, 2,
        #                                                          return_attn_weights=True,
        #                                                          attn_type='soft')

    def forward(self, x):
        out = self.encoder(x)
        out, att = self.classifier(out)
        return torch.cat((out, att), dim=-1)


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
        # out = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        out = self.model.avgpool(x)

        return out.squeeze()


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
        # out = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        out = x.squeeze()
        # out = self.model.fc(x)

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
