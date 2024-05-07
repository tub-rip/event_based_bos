import logging
import math
import cv2

from math import exp

import numpy as np
import scipy
import scipy.fftpack
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

logger = logging.getLogger(__name__)

SQRT_2PI = np.sqrt(2 * np.pi)


def gaussian_1d(mean, std, x):
    """Returns gaussian_1d coefficient (pdf value) for a given mean, std, x.
    Args:
        mean (scalar, or tensor) ... mean of the Gaussian dist.
        std (scalar, or tensor) ... Std (not Var) of the Gaussian dist.
        x (scalar, array, or tensor) ... x. the same dimension as mean.
    Returns:
        (torch.tensor) ... Gaussian coefficient.
    """
    if isinstance(x, (np.ndarray, float)):
        y = (x - mean) ** 2 / (2 * (std ** 2))
        Z = SQRT_2PI * std
        return np.exp(-y) / Z
    elif isinstance(x, torch.Tensor):
        y = (x - mean) ** 2 / (2 * (std ** 2))
        Z = torch.sqrt(torch.tensor(2 * np.pi)) * std
        return torch.exp(-y) / Z
    raise NotImplementedError


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


class SobelTorch(nn.Module):
    """Sobel operator for pytorch, for divergence calculation.
        This is equivalent implementation of
        ```
        sobelx = cv2.Sobel(flow[0], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(flow[1], cv2.CV_64F, 0, 1, ksize=3)
        dxy = (sobelx + sobely) / 8.0
        ```
    Args:
        ksize (int) ... Kernel size of the convolution operation.
        in_channels (int) ... In channles.
        cuda_available (bool) ... True if cuda is available.
    """

    def __init__(
            self, ksize: int = 3, in_channels: int = 2, cuda_available: bool = False, precision="32",
            padding: int=1, padding_mode="replicate"
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels
        self.filter_dx = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            padding_mode="replicate",
            bias=False,
        )
        self.filter_dy = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            padding_mode="replicate",
            bias=False,
        )

        assert ksize in [3, 5]
        # x in height direction
        if ksize == 3:
            Gx = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            Gy = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        else:
            Gx = torch.Tensor([
                [-2, -2, -4, -2, -2],
                [-1, -1, -2, -1, -1],
                [0, 0, 0, 0, 0],
                [1, 1, 2, 1, 1],
                [2, 2, 4, 2, 2]
            ])
            Gy = torch.Tensor([
                [-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2],
                [-4, -2, 0, 2, 4],
                [-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2]
            ])

        if precision == "64":
            Gx = Gx.double()
            Gy = Gy.double()

        if self.cuda_available:
            Gx = Gx.cuda()
            Gy = Gy.cuda()

        self.filter_dx.weight = nn.Parameter(Gx.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.filter_dy.weight = nn.Parameter(Gy.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, img):
        """
        Args:
            img (torch.Tensor) ... [b x (2 or 1) x H x W]. The 2 ch is [h, w] direction.

        Returns:
            sobel (torch.Tensor) ... [b x (4 or 2) x (H - 2) x (W - 2)].
                4ch means Sobel_x on xdim, Sobel_y on ydim, Sobel_x on ydim, and Sobel_y on xdim.
                To make it divergence, run `(sobel[:, 0] + sobel[:, 1]) / 8.0`.
        """
        if self.in_channels == 2:
            dxx = self.filter_dx(img[..., [0], :, :])
            dyy = self.filter_dy(img[..., [1], :, :])
            dyx = self.filter_dx(img[..., [1], :, :])
            dxy = self.filter_dy(img[..., [0], :, :])
            return torch.cat([dxx, dyy, dyx, dxy], dim=1)
        elif self.in_channels == 1:
            dx = self.filter_dx(img[..., [0], :, :])
            dy = self.filter_dy(img[..., [0], :, :])
            return torch.cat([dx, dy], dim=1)


def poisson_reconstruct(
        grady: np.ndarray, gradx: np.ndarray, boundarysrc: np.ndarray
) -> np.ndarray:
    """Run Poisson reconstruction.
    Code obtained from https://gist.github.com/jackdoerner/b9b5e62a4c3893c76e4c

    Args:
        grady (np.ndarray): [H, W] np array
        gradx (np.ndarray): [H, W] np array
        boundarysrc (np.ndarray): The boundary condition. Same shape as grady and gradx.

    Returns:
        np.ndarray: Poisson reconstruction (integration).
    """
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:, :-1] - grady[:-1, :-1]
    gxx = gradx[:-1, 1:] - gradx[:-1, :-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = (
            -4 * boundary[1:-1, 1:-1]
            + boundary[1:-1, 2:]
            + boundary[1:-1, 0:-2]
            + boundary[2:, 1:-1]
            + boundary[0:-2, 1:-1]
    )
    f = f[1:-1, 1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm="ortho")
    fsin = scipy.fftpack.dst(tt.T, norm="ortho").T

    # Eigenvalues
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = (2 * np.cos(math.pi * x / (f.shape[1] + 2)) - 2) + (
            2 * np.cos(math.pi * y / (f.shape[0] + 2)) - 2
    )

    f = fsin / denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm="ortho")
    img_tt = scipy.fftpack.idst(tt.T, norm="ortho").T

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] = img_tt

    return result


def strain_variant(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # if len(u.shape) == 2:
    #     u = u[None]
    # if len(v.shape) == 2:
    #     v = v[None]
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    du_dy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    dv_dx = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return du_dx ** 2 + dv_dy ** 2 + 0.5 * (du_dy + dv_dx) ** 2


# SSIM code from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


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


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
