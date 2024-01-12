"""
Created by: Liping Zhang
CUHK Lab of AI in Radiology (CLAIR)
Department of Imaging and Interventional Radiology
Faculty of Medicine
The Chinese University of Hong Kong (CUHK)
Email: lpzhang@link.cuhk.edu.hk
Copyright (c) CUHK 2023.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        # data_range = data_range[:, None, None, None]
        data_range = data_range.reshape(data_range.shape + (1,)*(len(X.shape)-len(data_range.shape)))

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        out_channels, in_channels = X.shape[1], X.shape[1]
        ux = F.conv2d(X, self.w.expand(out_channels,-1,-1,-1), groups=in_channels)
        uy = F.conv2d(Y, self.w.expand(out_channels,-1,-1,-1), groups=in_channels)
        uxx = F.conv2d(X * X, self.w.expand(out_channels,-1,-1,-1), groups=in_channels)
        uyy = F.conv2d(Y * Y, self.w.expand(out_channels,-1,-1,-1), groups=in_channels)
        uxy = F.conv2d(X * Y, self.w.expand(out_channels,-1,-1,-1), groups=in_channels)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()
