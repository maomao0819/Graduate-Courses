import torch
import torch.nn as nn
import torch.nn.functional as F
import wavencoder

from config import Config

cfg = Config()


class audio_net(nn.Module):
    def __init__(self):
        super(audio_net, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained = True)
        self.classifier = wavencoder.models.LSTM_Attn_Classifier(512, 32, 2,
                                                                return_attn_weights=True,
                                                                attn_type='soft')        

    def forward(self, x):
        z = self.encoder(x)
        out, _ = self.classifier(z)
        return out

