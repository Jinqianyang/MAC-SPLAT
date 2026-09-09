# utils/compute_ssim.py
import torch
import torch.nn.functional as F
from torch import nn
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, data_range=1.0, size_average=True, full=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean([1, 2, 3])

    if full:
        # This is for a different functionality, we'll return ssim_map for our use case
        # For simplicity, if full is True, we return the spatial map.
        return ssim_map

    return ret

def compute_ssim(img1, img2, window_size=11, data_range=1.0, size_average=True, full=False):
    # This function is now just a wrapper
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, data_range, size_average, full)


class DSSIM(nn.Module):
    def __init__(self, window_size=11, data_range=1.0):
        super(DSSIM, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.channel = None
        self.window = None

    def forward(self, img1, img2):
        # DSSIM = (1 - SSIM) / 2
        # We want the spatial map of losses, so size_average=False, and full=True (in our modified ssim)
        (_, channel, _, _) = img1.size()
        if self.channel is None or self.channel != channel:
            self.window = create_window(self.window_size, channel)
            self.channel = channel
        
        window = self.window.to(img1.device).type_as(img1)
        
        # We need the spatial map of SSIM, so size_average=False
        ssim_map = _ssim(img1, img2, window, self.window_size, self.channel, self.data_range, size_average=False, full=True)
        
        # The result should be per-pixel dssim loss
        dssim_map = (1.0 - ssim_map) / 2.0
        
        # The reduction (mean or sum) will be handled outside in the main loss calculation
        return dssim_map