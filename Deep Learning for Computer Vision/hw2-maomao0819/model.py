
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

"""
P1 - DCGAN
reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()

        n_latent = 100
        n_feature_map = 64
        n_channel = 3

        def generator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            block = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            ]
            return block

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            *generator_block(n_latent, n_feature_map * 8, stride=1, padding=0),
            # state size. (n_feature_map x 8) x 4 x 4
            *generator_block(n_feature_map * 8, n_feature_map * 4),
            # state size. (n_feature_map x 4) x 8 x 8
            *generator_block(n_feature_map * 4, n_feature_map * 2),
            # state size. (n_feature_map x 2) x 16 x 16
            *generator_block(n_feature_map * 2, n_feature_map),
            # state size. (n_feature_map) x 32 x 32
            nn.ConvTranspose2d(n_feature_map, n_channel, 4, 2, 1, bias=False),
            # state size. (n_channel) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()

        n_channel = 3
        n_feature_map = 64
        self.n_feature_map = n_feature_map

        # image_size -> (image_size + 2 * padding - kernel_size) / (stride) + 1
        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return block

        self.feature_extract = nn.Sequential(
            # input is (n_channel) x 64 x 64
            nn.Conv2d(n_channel, n_feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_feature_map) x 32 x 32
            *discriminator_block(n_feature_map, n_feature_map * 2),
            # state size. (n_feature_map x 2) x 16 x 16
            *discriminator_block(n_feature_map * 2, n_feature_map * 4),
            # state size. (n_feature_map x 4) x 8 x 8
            *discriminator_block(n_feature_map * 4, n_feature_map * 8),
            # state size. (n_feature_map x 8) x 4 x 4
            nn.Conv2d(n_feature_map * 8, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, input):
        hidden = self.feature_extract(input)
        return hidden


class My_Generator(nn.Module):
    def __init__(self, n_latent=100, n_feature_map=64, n_channel=3):
        super(My_Generator, self).__init__()

        def generator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            block = [
                nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            ]
            return block

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            *generator_block(n_latent, n_feature_map * 8, stride=1, padding=0),
            # state size. (n_feature_map x 8) x 4 x 4
            *generator_block(n_feature_map * 8, n_feature_map * 4),
            # state size. (n_feature_map x 4) x 8 x 8
            *generator_block(n_feature_map * 4, n_feature_map * 2),
            # state size. (n_feature_map x 2) x 16 x 16
            *generator_block(n_feature_map * 2, n_feature_map),
            # state size. (n_feature_map) x 32 x 32
            nn.ConvTranspose2d(n_feature_map, n_channel, 4, 2, 1, bias=False),
            # state size. (n_channel) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class My_Discriminator(nn.Module):
    def __init__(self, n_channel=3, n_feature_map=64):
        super(My_Discriminator, self).__init__()

        self.n_feature_map = n_feature_map

        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            block = [
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return block

        self.feature_extract = nn.Sequential(
            # input is (n_channel) x 64 x 64
            nn.Conv2d(n_channel, n_feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_feature_map) x 32 x 32
            *discriminator_block(n_feature_map, n_feature_map * 2),
            # state size. (n_feature_map x 2) x 16 x 16
            *discriminator_block(n_feature_map * 2, n_feature_map * 4),
            # state size. (n_feature_map x 4) x 8 x 8
            *discriminator_block(n_feature_map * 4, n_feature_map * 8),
            # state size. (n_feature_map x 8) x 4 x 4
            nn.Conv2d(n_feature_map * 8, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.feature_extract(input)
        return out

"""
P2 - Diffusion DPMM
reference: https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/tree/main/DiffusionFreeGuidence
"""

# main model

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, time_length, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        embedding = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        embedding = torch.exp(-embedding)
        position = torch.arange(time_length).float()
        embedding = position[:, None] * embedding[None, :]
        assert list(embedding.shape) == [time_length, d_model // 2]
        embedding = torch.stack([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        assert list(embedding.shape) == [time_length, d_model // 2, 2]
        embedding = embedding.view(time_length, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embedding, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, time_step):
        embedding = self.time_embedding(time_step)
        return embedding


class ConditionalEmbedding(nn.Module):
    def __init__(self, n_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.conditional_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=n_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, time_step):
        embedding = self.conditional_embedding(time_step)
        return embedding


class DownSample(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, in_channel, 3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)

    def forward(self, x, time_embedding, condition_embedding):
        x = self.conv_1(x) + self.conv_2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.convtrans = nn.ConvTranspose2d(in_channel, in_channel, 5, 2, 2, 1)

    def forward(self, x, time_embedding, condition_embedding):
        x = self.convtrans(x)
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channel)
        self.proj_q = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        hidden = self.group_norm(x)
        query = self.proj_q(hidden)
        key = self.proj_k(hidden)
        value = self.proj_v(hidden)

        query = query.permute(0, 2, 3, 1).view(B, H * W, C)
        key = key.view(B, C, H * W)
        w = torch.bmm(query, key) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        value = value.permute(0, 2, 3, 1).view(B, H * W, C)
        hidden = torch.bmm(w, value)
        assert list(hidden.shape) == [B, H * W, C]
        hidden = hidden.view(B, H, W, C).permute(0, 3, 1, 2)
        hidden = self.proj(hidden)

        return x + hidden

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, tdim, dropout, attention=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        )
        self.time_embedding_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channel),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channel),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attention:
            self.attention = AttentionBlock(out_channel)
        else:
            self.attention = nn.Identity()


    def forward(self, x, time_embedding, labels):
        hidden = self.block1(x)
        hidden += self.time_embedding_proj(time_embedding)[:, :, None, None]
        hidden += self.cond_proj(labels)[:, :, None, None]
        hidden = self.block2(hidden)

        hidden = hidden + self.shortcut(x)
        out = self.attention(hidden)
        return out


class UNet(nn.Module):
    def __init__(self, time_length, n_labels, channel, channel_multiply, n_residual_blocks, dropout):
        super().__init__()
        tdim = channel * 4
        self.time_embedding = TimeEmbedding(time_length, channel, tdim)
        self.cond_embedding = ConditionalEmbedding(n_labels, channel, tdim)
        self.head = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        channels = [channel]  # record output channel when dowmsample for upsample
        now_channel = channel
        for i, multiply in enumerate(channel_multiply):
            out_channel = channel * multiply
            for _ in range(n_residual_blocks):
                self.downblocks.append(ResidualBlock(in_channel=now_channel, out_channel=out_channel, tdim=tdim, dropout=dropout))
                now_channel = out_channel
                channels.append(now_channel)
            if i != len(channel_multiply) - 1:
                self.downblocks.append(DownSample(now_channel))
                channels.append(now_channel)

        self.middleblocks = nn.ModuleList([
            ResidualBlock(now_channel, now_channel, tdim, dropout, attention=True),
            ResidualBlock(now_channel, now_channel, tdim, dropout, attention=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, multiply in reversed(list(enumerate(channel_multiply))):
            out_channel = channel * multiply
            for _ in range(n_residual_blocks + 1):
                self.upblocks.append(ResidualBlock(in_channel=channels.pop() + now_channel, out_channel=out_channel, tdim=tdim, dropout=dropout, attention=False))
                now_channel = out_channel
            if i != 0:
                self.upblocks.append(UpSample(now_channel))
        assert len(channels) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_channel),
            Swish(),
            nn.Conv2d(now_channel, 3, 3, stride=1, padding=1)
        )
 

    def forward(self, image, time_length, labels):
        time_embedding = self.time_embedding(time_length)
        # batch size x 512
        condition_embedding = self.cond_embedding(labels)
        # batch size x 512

        # Downsampling
        hidden = self.head(image)
        hiddens = [hidden]
        for layer in self.downblocks:
            hidden = layer(hidden, time_embedding, condition_embedding)
            hiddens.append(hidden)
        # Middle
        for layer in self.middleblocks:
            hidden = layer(hidden, time_embedding, condition_embedding)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                hidden = torch.cat([hidden, hiddens.pop()], dim=1)
            hidden = layer(hidden, time_embedding, condition_embedding)
        hidden = self.tail(hidden)

        assert len(hiddens) == 0
        return hidden

# diffusion

def extract(v, time_step, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = time_step.device
    out = torch.gather(v, index=time_step, dim=0).float().to(device)
    return out.view([time_step.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, time_length):
        super().__init__()

        self.model = model
        self.time_length = time_length

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, time_length).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, image, labels):
        """
        Algorithm 1.
        """
        time = torch.randint(self.time_length, size=(image.shape[0], ), device=image.device)
        noise = torch.randn_like(image)
        image_t =  extract(self.sqrt_alphas_bar, time, image.shape) * image + \
            extract(self.sqrt_one_minus_alphas_bar, time, image.shape) * noise
        loss = F.mse_loss(self.model(image_t, time, labels), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, time_length, w = 0., grid_image=False):
        super().__init__()

        self.model = model
        self.time_length = time_length
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, time_length).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:time_length]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, image_t, time_step, eps):
        assert image_t.shape == eps.shape
        return extract(self.coeff1, time_step, image_t.shape) * image_t - extract(self.coeff2, time_step, image_t.shape) * eps

    def p_mean_variance(self, image_t, time_step, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, time_step, image_t.shape)
        eps = self.model(image_t, time_step, labels)
        nonEps = self.model(image_t, time_step, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(image_t, time_step, eps=eps)
        return xt_prev_mean, var

    def forward(self, image_T, labels):
        """
        Algorithm 2.
        """
        image_t = image_T
        time_pbar = tqdm(reversed(range(self.time_length)), desc="Time", total=self.time_length)
        for time_step in time_pbar:
            # time_pbar.set_description(f"Time [{self.time_length - time_step}/{self.time_length}]")
            time = image_t.new_ones([image_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(image_t=image_t, time_step=time, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(image_t)
            else:
                noise = 0
            image_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(image_t).int().sum() == 0, "nan in tensor."
        image = image_t
        return torch.clip(image, -1, 1) 


"""
P3 - DANN
reference: https://github.com/fungtion/DANN
"""

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN_model(nn.Module):
    def __init__(self, n_class=10):
        super(DANN_model, self).__init__()

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            # state size. 64 x 24 x 24
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # state size. 64 x 12 x 12
            nn.Conv2d(64, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            # state size. 48 x 8 x 8
            nn.Dropout2d(p=0.3, inplace=True),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # state size. 48 x 4 x 4
        )

        self.class_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(48 * 4 * 4, 192),
            nn.BatchNorm1d(192),
            nn.Mish(True),
            nn.Dropout(0.3),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.Mish(True),
            nn.Linear(128, n_class),
        )

        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(48 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.Mish(True),
            nn.Linear(128, 2),
        )

    def forward(self, input, alpha=1):
        feature = self.feature_extract(input)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        output = {}
        output["class"] = class_output
        output["domain"] = domain_output
        return output

class ExtractedDANN(DANN_model):
    def __init__(self, n_class=10):
        super().__init__(n_class)

    def forward(self, input, alpha):
        feature = self.feature_extract(input)
        feature = feature.view(-1, 48 * 4 * 4)
        return feature