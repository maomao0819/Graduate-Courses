import torch
import torch.nn as nn
import torchvision 

class Resnet50Model(nn.Module):
    def __init__(self, n_classes=65, fix_backbone=False):
      super(Resnet50Model, self).__init__()
      self.backbone = torchvision.models.resnet50(pretrained=False)
      if fix_backbone:
        for child in self.backbone.children():
          for param in child.parameters():
            param.requires_grad = False
      self.classifier = nn.Linear(1000, n_classes)

    def forward(self, input):
      feature = self.backbone(input)
      output = self.classifier(feature)
      return output