#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from utils.image_utils import pixels_to_bayer_mask

def l1_loss_mean(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt): 
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


''' RawSplats Loss Utils'''
def huber_loss(render, gt, delta, weights):
    l1_err = torch.abs(render - gt)
    quad_loss = 0.5 * (l1_err ** 2)
    linear_loss = delta * (l1_err - 0.5 * delta)

    weighted_quad_loss = quad_loss * weights
    weighted_linear_loss = linear_loss * weights
    loss = (torch.where(l1_err <= delta, weighted_quad_loss, weighted_linear_loss)).mean()
    return loss


def loss_fn(render, gt, bayer_mask, opt):
    # apply bayer mask to the ground truth
    bayer_mask = torch.from_numpy(bayer_mask.astype(np.uint8)).permute(2, 0, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bayer_mask = bayer_mask.to(device)

    loss_weight = torch.broadcast_to(bayer_mask, gt.shape)


    # gamma curve to weight errors in dark regions more
    scaling_grad = 1. / (1e-3 + render.detach())
    Ll1 = l1_loss(render, gt)
    loss_reg = Ll1 * scaling_grad
    
    loss = (1 - opt.lambda_dssim) * (Ll1.mean()) + opt.lambda_dssim * (loss_reg * loss_weight).mean()

    return loss
