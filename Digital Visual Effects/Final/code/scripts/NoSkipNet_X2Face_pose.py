# Pix2Pix : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, input_pose, audio, expr, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], inner_nc=512):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        # netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,
        #                      use_dropout=use_dropout, gpu_ids=gpu_ids)
        netG = UnetGeneratorBetterUpsampler(input_nc, output_nc, input_pose, audio, expr, 7, ngf,
                                            inner_nc=inner_nc, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGeneratorBetterUpsampler(input_nc, output_nc, input_pose, audio, expr, 8, ngf,
                                            inner_nc=inner_nc, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type=init_type)
    return netG


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Defines the SkipNetWork


class Pix2PixModel(nn.Module):
    def __init__(self, input_nc, output_nc, input_pose=False, audio=False, expr=False, inner_nc=512):
        super(Pix2PixModel, self).__init__()

        self.netG = define_G(input_nc, output_nc, input_pose, audio, expr, 64,
                             'unet_128', 'batch', False, 'xavier', [0], inner_nc=inner_nc)

    def forward(self, *cycles):
        # First one
        xc = self.netG(cycles[0], *cycles[1:])
        return xc


class UnetGeneratorBetterUpsampler(nn.Module):
    def __init__(self, input_nc, output_nc, input_pose, audio, expr, num_downs, ngf=64, inner_nc=512,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorBetterUpsampler, self).__init__()
        self.gpu_ids = gpu_ids
        # construct unet structure
        unet_block = UnetSkipConnectionBlockBetterUpsampler(
            ngf * 8, inner_nc, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, input_pose=input_pose, audio=audio, expr=expr)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockBetterUpsampler(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, input_pose=input_pose, audio=audio, expr=expr)
        self.model = unet_block

    def forward(self, x, *views):
        # if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
        #     output, output_orig = nn.parallel.data_parallel(self.model, (x, views), self.gpu_ids)
        #     return output, output_orig
        # else:
        return self.model(x, *views)


class UnetSkipConnectionBlockBetterUpsampler(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, input_pose=False, audio=False, expr=False):
        super(UnetSkipConnectionBlockBetterUpsampler, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc, outer_nc,
                               kernel_size=3, stride=1,
                               padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, nn.Tanh()]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc, outer_nc,
                               kernel_size=3, stride=1,
                               padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
            if input_pose is True:
                self.posefrombottle = nn.Linear(128, 3)
                self.bottlefrompose = nn.Sequential(nn.Linear(3, 128, bias=False), nn.BatchNorm1d(128))
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc, outer_nc,
                               kernel_size=3, stride=1,
                               padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
            self.dropout = nn.Dropout(0.5)
        self.submodule = submodule

    def forward(self, x_orig, *other_inputs):
        x_fv = self.down(x_orig)
        others_fv = []
        others_fv = other_inputs
        if self.innermost:
            if other_inputs is not None:
                if not len(others_fv) == 0:
                    x_fv_bn = x_fv.detach()
                    pose = self.posefrombottle(x_fv_bn.squeeze().unsqueeze(0))
                    bn_from_pose = self.bottlefrompose(pose)
                    bn_from_target = self.bottlefrompose(others_fv[0][0])
                    x_fv = x_fv_bn + (-bn_from_pose + bn_from_target).unsqueeze(2).unsqueeze(3).detach()
            x = self.up(x_fv)
            return x, x_fv, others_fv
        if self.outermost:
            x, x_fv, _ = self.submodule(x_fv, *others_fv)
            x = self.up(x)
            return x, x_fv, others_fv
        else:
            x, x_fv, _ = self.submodule(x_fv, *others_fv)
            if self.use_dropout:
                x = self.dropout(self.up(x))
            else:
                x = self.up(x)

            return x, x_fv, others_fv


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.innermost:
            xc_orig = self.model[2](self.model[1](self.model[0](x)))
            x_new = self.model[3](xc_orig)
            x_new = self.model[4](x_new)
            self.model.fc = xc_orig
            return torch.cat([x, x_new], 1)
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x,  self.model(x)], 1)
