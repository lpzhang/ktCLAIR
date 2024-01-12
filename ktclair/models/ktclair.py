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

import math
from typing import List, Tuple, Optional, Callable, Sequence, Union

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet3d import Unet3D
from einops import rearrange


class NormUnet3D(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        bg_drop_prob: float = None,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet3D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

        self.patch_size = 2**num_pools - 1
        self.bg_drop_prob = bg_drop_prob

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 5, 1, 2, 3, 4).reshape(b, 2 * c, d, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, d, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, d, h, w).permute(0, 2, 3, 4, 5, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, d, h, w = x.shape
        # x = x.view(b, 2, c // 2 * d * h * w)

        # # mean = x.mean(dim=2).view(b, 2, 1, 1, 1)
        # # std = x.std(dim=2).view(b, 2, 1, 1, 1)
        # mean = x.mean(dim=2).repeat(1,1,c//2).view(b, c, 1, 1, 1)
        # std = x.std(dim=2).repeat(1,1,c//2).view(b, c, 1, 1, 1)

        # x = x.view(b, c, d, h, w)

        mean = x.mean(dim=[-3,-2,-1]).view(b, c, 1, 1, 1)
        std = x.std(dim=[-3,-2,-1]).view(b, c, 1, 1, 1)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, d, h, w = x.shape
        # w_mult = ((w - 1) | 15) + 1
        # h_mult = ((h - 1) | 15) + 1
        w_mult = ((w - 1) | self.patch_size) + 1
        h_mult = ((h - 1) | self.patch_size) + 1
        d_mult = ((d - 1) | 0) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        d_pad = [math.floor((d_mult - d) / 2), math.ceil((d_mult - d) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad + d_pad)

        return x, (d_pad, h_pad, w_pad, d_mult, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        d_pad: List[int],
        h_pad: List[int],
        w_pad: List[int],
        d_mult: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., d_pad[0] : d_mult - d_pad[1], h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def bg_drop(self, x: torch.Tensor, bg_drop_prob: float) -> torch.Tensor:
        ''' x (b c d h w comp)
        '''
        # only drop bg in the sx-dimension here refers as w-dimension
        data1d_sx = fastmri.complex_abs(x) # (b c d h w)
        data1d_sx = data1d_sx.mean(dim=(0,1,2,3)) # (w)
        k = max(int(round((1 - bg_drop_prob) * data1d_sx.numel())), 0)
        threshold = torch.min(torch.topk(data1d_sx, k).values)
        data1d_sx = data1d_sx >= threshold
        data1d_sx = data1d_sx.nonzero(as_tuple=True)[0]
        left_sx = data1d_sx[0] # [
        right_sx = data1d_sx[-1] + 1 # )

        x = x[...,left_sx:right_sx,:]

        return x, left_sx, right_sx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, c, d, h, w, comp
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # remove bg area
        if self.bg_drop_prob is not None:
            shortcut = x
            x, left_sx, right_sx = self.bg_drop(x, self.bg_drop_prob)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        # add bg area back to maintain image size
        if self.bg_drop_prob is not None:
            shortcut[...,left_sx:right_sx,:] = x
            x = shortcut

        return x


class Sensitivity3DModule(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.mask_center = mask_center
        self.norm_unet = NormUnet3D(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )
        self._coil_dim = 1
        self._spatial_dims = (3, 4)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, d, h, w, comp = x.shape

        return x.view(b * c, 1, d, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, d, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, d, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=self._coil_dim).unsqueeze(-1).unsqueeze(self._coil_dim)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # b, c, d, h, w, comp
        if self.mask_center:
            masked_kspace = masked_kspace * mask + 0.0

        # convert to image space
        images, batches = self.chans_to_batch_dim(
            self.backward_operator(masked_kspace, dim=self._spatial_dims)
        )

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class KSpaceSequentialModule(nn.Module):
    """
    Implements KSpaceSequentialModule
    """
    def __init__(self, models: nn.ModuleList):
        super().__init__()
        self.models = models

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input 5D/6D tensor of shape `(N, in_chans, H, W, 2)`
                or `(N, in_chans, D, H, W, 2)`.
            mask: Input 6D/7D tensor of shape `(N, S, 1, H, W, 1)`
                or `(N, S, 1, D, H, W, 1)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W, 2)`
                or `(N, out_chans, D, H, W, 2)`.
        """
        # assert len(self.models) == mask.shape[1]
        if len(self.models) != mask.shape[1]:
            raise ValueError(
                f"Number of models should be equal to number of masks."
                f"Got {len(self.models)} and {mask.shape[1]}."
            )
        for i, model in enumerate(self.models):
            x = model(x, mask[:,i])
        return x


class KSpaceModule3D(nn.Module):
    """
    Implements KSpaceModule3D for arbitrary models
    """
    def __init__(
        self,
        model: nn.Module,
        conv_mode: str = 'real',
        physics: bool = True,
        normalized: bool = True,
    ):
        """
        Args:
            model: Arbitrary model
            conv_mode: "real" / "complex"
        """
        super().__init__()
        self.model = model
        self.conv_mode = conv_mode
        self.physics = physics
        self.normalized = normalized

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input 6D tensor of shape `(N, in_chans, D, H, W, 2)`.
            mask: Input 6D tensor of shape `(N, 1, D, H, W, 1)`.

        Returns:
            Output tensor of shape `(N, out_chans, D, H, W, 2)`.
        """
        b, c, d, h, w, comp = x.shape
        if not comp == 2:
            raise ValueError("Tensor does not have separate complex dim.")

        # normalize
        if self.normalized:
            mean = x.mean(dim=[-5,-4,-3,-2]).view(b, 1, 1, 1, 1, comp)
            std = x.std(dim=[-5,-4,-3,-2]).view(b, 1, 1, 1, 1, comp)
            x = (x - mean) / std

        if self.physics:
            shortcut = x

        if self.conv_mode != 'complex':
            x = rearrange(x, 'b c d h w comp -> b (c comp) d h w', comp=2)
        else:
            x = torch.view_as_complex(x)

        x = self.model(x)

        if self.conv_mode != 'complex':
            x = rearrange(x, 'b (c comp) d h w -> b c d h w comp', comp=2)
        else:
            x = torch.view_as_real(x)

        if self.physics:
            x = torch.where(mask, x, shortcut)

        # unnormalize
        if self.normalized:
            x = x * std + mean

        return x


class KSpaceKernel3D(nn.Module):
    """
    Implements KSpaceKernel3D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        conv_mode: str = 'real',
        acti_func: str = None,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            conv_mode (string, optional): 'complex' or 'real'. Default: 'real'
            acti_func (string, optional): activation function
            kwargs: Additional arguments for the convolution
        """
        super(KSpaceKernel3D, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv3d(
                in_channels, out_channels, kernel_size, **kwargs,
            ) for _ in range(1 if conv_mode != 'complex' else 2)]
        )

        if acti_func == 'ReLU':
            self.acti_func = nn.ReLU()
        elif acti_func == 'LeakyReLU':
            self.acti_func = nn.LeakyReLU(inplace=True)
        else:
            self.acti_func = None

    def forward(self, x):
        """
        Args:
            x: Input 5D tensor of shape `(N, in_chans, D, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, D, H, W)`.
        """
        if not torch.is_complex(x):
            x = self.convs[0](x)
            if self.acti_func is not None:
                x = self.acti_func(x)
        else:
            real = self.convs[0](x.real) - self.convs[1](x.imag)
            imag = self.convs[0](x.imag) + self.convs[1](x.real)
            if self.acti_func is not None:
                real = self.acti_func(real)
                imag = self.acti_func(imag)
            x = real.type(torch.complex64) + 1j*imag.type(torch.complex64)

        return x


class XFBlockModule(nn.Module):
    def __init__(self,
        forward_operator: Callable,
        backward_operator: Callable,
        model: nn.Module,
        dc_mode: str = 'GD',
        divide_by_n: bool = False,
        use_baseline: bool = True,
    ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.dc_weight = nn.Parameter(torch.ones(1))
        self._coil_dim = 1
        self._temporal_dim = 2
        self._spatial_dims = (3, 4)
        self.divide_by_n = divide_by_n
        self.dc_mode = dc_mode
        self.use_baseline = use_baseline

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return self.forward_operator(fastmri.complex_mul(x, sens_maps), dim=self._spatial_dims)

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            self.backward_operator(x, dim=self._spatial_dims), fastmri.complex_conj(sens_maps)
        ).sum(dim=self._coil_dim, keepdim=True)

    def to_xf_space(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None,
    ):
        """
        transform to x-f space with subtraction of average temporal frame in multi-coil setting
        6D data (b c d h w comp) with the `d' is the temporal dimension
        kspace (b c d h w comp)
        masked_kspace (b c d h w comp)
        mask (b 1 1 h w 1)
        sens_maps (b c d h w comp)
        """
        if self.divide_by_n:
            kspace_avg = kspace.mean(dim=self._temporal_dim, keepdim=True) # (b c 1 h w comp)
        else:
            kspace_avg = torch.div(
                masked_kspace.sum(dim=self._temporal_dim, keepdim=True),
                torch.clamp(mask.sum(dim=self._temporal_dim, keepdim=True), min=1)
            ) # (b c 1 h w comp)

        # subtract the temporal average frame
        kspace_diff = kspace - kspace_avg # (b c d h w comp)

        # the coil combined temporal difference in image space
        x_t_diff = self.sens_reduce(kspace_diff, sens_maps) # (b 1 d h w comp)

        # the coil combined temporal average frame in image space
        x_t_avg = self.sens_reduce(kspace_avg, sens_maps) # (b 1 1 h w comp)

        # transform to x-f space to get the baseline
        x_f_diff = self.forward_operator(x_t_diff, dim=[self._temporal_dim]) # (b 1 d h w comp)
        x_f_avg = self.forward_operator(x_t_avg, dim=[self._temporal_dim]) # (b 1 1 h w comp)

        return x_f_diff, x_f_avg

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        # # transform to x-f space to get the baseline
        # x_f_diff, x_f_avg = self.to_xf_space(
        #     current_kspace, ref_kspace, mask, sens_maps
        # ) # (b 1 d h w comp) (b 1 1 h w comp)

        # model_term = self.sens_expand(
        #     self.backward_operator(
        #         self.model(x_f_diff) + x_f_avg, dim=[self._temporal_dim]
        #     ),
        #     sens_maps
        # ) # (b 1 d h w comp)

        # b, c, d, h, w, comp
        if not self.use_baseline:
            xt = self.sens_reduce(current_kspace, sens_maps)
            # transform to x-f space
            xf = self.forward_operator(xt, dim=[self._temporal_dim])
            xf = self.model(xf)
        else:
            # transform to x-f space to get the baseline
            x_f_diff, x_f_avg = self.to_xf_space(
                current_kspace, ref_kspace, mask, sens_maps
            ) # (b 1 d h w comp) (b 1 1 h w comp)
            xf = self.model(x_f_diff) + x_f_avg

        model_term = self.sens_expand(
            self.backward_operator(xf, dim=[self._temporal_dim]), sens_maps
        ) # (b 1 d h w comp)

        if self.dc_mode == 'GD':
            zero = torch.zeros_like(current_kspace)
            soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
            return current_kspace - soft_dc - model_term
        else:
            return model_term


class XTBlockModule(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self,
        forward_operator: Callable,
        backward_operator: Callable,
        model: nn.Module,
        dc_mode: str = 'GD',
        use_xf: bool = False,
    ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.dc_weight = nn.Parameter(torch.ones(1))
        self._coil_dim = 1
        self._spatial_dims = (3, 4)
        self.dc_mode = dc_mode
        self._temporal_dim = 2
        self.use_xf = use_xf

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return self.forward_operator(fastmri.complex_mul(x, sens_maps), dim=self._spatial_dims)

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            self.backward_operator(x, dim=self._spatial_dims), fastmri.complex_conj(sens_maps)
        ).sum(dim=self._coil_dim, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        # b, c, d, h, w, comp
        xt = self.sens_reduce(current_kspace, sens_maps)

        # transform to x-f space
        if self.use_xf:
            xf = self.forward_operator(xt, dim=[self._temporal_dim])
            xt = torch.cat((xt, xf), dim=self._coil_dim)

        model_term = self.model(xt)

        if self.use_xf:
            xt, xf = model_term.chunk(2, dim=self._coil_dim)
            xf = self.backward_operator(xf, dim=[self._temporal_dim])
            model_term = (xt + xf) / 2.

        model_term = self.sens_expand(model_term, sens_maps)
        # model_term = self.sens_expand(
        #     self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        # )

        if self.dc_mode == 'GD':
            zero = torch.zeros_like(current_kspace)
            soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
            return current_kspace - soft_dc - model_term
        else:
            return model_term


class KTCLAIR(nn.Module):
    """KTCLAIR
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        ### sens_model
        sens_model: str = "Sensitivity3DModule",
        sens_chans: int = 8,
        sens_pools: int = 4,
        mask_center: bool = True,
        ### xt_model
        xt_model: str = "NormUnet3D",
        xt_num_cascades: int = 12,
        xt_inp_channels: int =2,
        xt_out_channels: int =2,
        xt_chans: int = 48,
        xt_pools: int = 4,
        xt_dc_mode: str = 'GD',
        xt_no_parameter_sharing: bool = True,
        xt_bg_drop_prob: float = None,
        ### kt_model
        kt_model: str = "KSpaceModule3D",
        kt_num_cascades: int = 12,
        kt_num_blocks: int = 4,
        kt_inp_channels: int = 10,
        kt_out_channels: int = 10,
        kt_conv_mode: str = 'complex',
        kt_kernel_sizes: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        kt_paddings: Union[int, Tuple[int, int, int]]  = (1, 1, 1),
        kt_acti_func: str = "none",
        kt_no_parameter_sharing: bool = True,
        kt_normalize: bool = True,
        ### xf_model
        xf_model: str = "NormUnet3D",
        xf_num_cascades: int = 12,
        xf_inp_channels: int =2,
        xf_out_channels: int =2,
        xf_chans: int = 48,
        xf_pools: int = 4,
        xf_dc_mode: str = 'GD',
        xf_no_parameter_sharing: bool = True,
        xf_bg_drop_prob: float = None,
        ### kf_model
        kf_model: str = "KSpaceModule3D",
        kf_num_cascades: int = 12,
        kf_num_blocks: int = 4,
        kf_inp_channels: int = 10,
        kf_out_channels: int = 10,
        kf_conv_mode: str = 'complex',
        kf_kernel_sizes: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        kf_paddings: Union[int, Tuple[int, int, int]]  = (1, 1, 1),
        kf_acti_func: str = "none",
        kf_no_parameter_sharing: bool = True,
        kf_normalize: bool = True,
    ):
        super().__init__()

        # sensitivity map
        if sens_model == "Sensitivity3DModule":
            self.sens_net = Sensitivity3DModule(
                forward_operator=forward_operator,
                backward_operator=backward_operator,
                chans=sens_chans,
                num_pools=sens_pools,
                mask_center=mask_center,
            )
        else:
            raise

        # num_cascades
        num_cascades = 0
        for name, num in zip(
            (xt_model, kt_model, xf_model, kf_model),
            (xt_num_cascades, kt_num_cascades, xf_num_cascades, kf_num_cascades)
        ):
            num_cascades = num_cascades if name.lower() == 'none' else max(num_cascades, num)

        # xt_model
        xt_cascades = [
            None for _ in range((num_cascades - xt_num_cascades) if xt_no_parameter_sharing else 0)
        ]
        if xt_model == "NormUnet3D":
            xt_cascades.extend([
                XTBlockModule(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    model=NormUnet3D(
                        in_chans=xt_inp_channels,
                        out_chans=xt_out_channels,
                        chans=xt_chans,
                        num_pools=xt_pools,
                        bg_drop_prob=xt_bg_drop_prob,
                    ),
                    dc_mode=xt_dc_mode,
                    use_xf=False,
                ) for _ in range(xt_num_cascades if xt_no_parameter_sharing else 1)
            ])
        elif xt_model == "NormUnet3DXF":
            xt_cascades.extend([
                XTBlockModule(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    model=NormUnet3D(
                        in_chans=xt_inp_channels * 2,
                        out_chans=xt_out_channels * 2,
                        chans=xt_chans,
                        num_pools=xt_pools,
                        bg_drop_prob=xt_bg_drop_prob,
                    ),
                    dc_mode=xt_dc_mode,
                    use_xf=True,
                ) for _ in range(xt_num_cascades if xt_no_parameter_sharing else 1)
            ])
        elif xt_model == "none":
            xt_cascades.extend([None for _ in range(xt_num_cascades if xt_no_parameter_sharing else 1)])
        else:
            raise
        self.xt_cascades = nn.ModuleList(xt_cascades)

        # kt_model
        kt_cascades = [
            None for _ in range((num_cascades - kt_num_cascades) if kt_no_parameter_sharing else 0)
        ]
        if kt_model == "KSpaceModule3D":
            kt_cascades.extend([
                KSpaceSequentialModule(
                    nn.ModuleList([
                        KSpaceModule3D(
                            model=nn.Sequential(*[
                                KSpaceKernel3D(
                                    in_channels=kt_inp_channels * 2 if kt_conv_mode != 'complex' else kt_inp_channels,
                                    out_channels=kt_out_channels * 2 if kt_conv_mode != 'complex' else kt_out_channels,
                                    conv_mode=kt_conv_mode,
                                    kernel_size=kt_kernel_sizes,
                                    padding=kt_paddings,
                                    acti_func=kt_acti_func,
                                    bias=False,
                                ) for i in range(kt_num_blocks)
                            ]),
                            conv_mode=kt_conv_mode,
                            physics=True,
                            normalized=kt_normalize,
                        )
                    ])
                ) for _ in range(kt_num_cascades if kt_no_parameter_sharing else 1)
            ])
        elif kt_model == "none":
            kt_cascades.extend([None for _ in range(kt_num_cascades if kt_no_parameter_sharing else 1)])
        else:
            raise
        self.kt_cascades = nn.ModuleList(kt_cascades)

        # xf_model
        xf_cascades = [
            None for _ in range((num_cascades - xf_num_cascades) if xf_no_parameter_sharing else 0)
        ]
        if xf_model == "NormUnet3D":
            xf_cascades.extend([
                XFBlockModule(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    model=NormUnet3D(
                        in_chans=xf_inp_channels,
                        out_chans=xf_out_channels,
                        chans=xf_chans,
                        num_pools=xf_pools,
                        bg_drop_prob=xf_bg_drop_prob,
                    ),
                    dc_mode=xf_dc_mode,
                    divide_by_n=False,
                    use_baseline=False,
                ) for _ in range(xf_num_cascades if xf_no_parameter_sharing else 1)
            ])
        elif xf_model == "none":
            xf_cascades.extend([None for _ in range(xf_num_cascades if xf_no_parameter_sharing else 1)])
        else:
            raise
        self.xf_cascades = nn.ModuleList(xf_cascades)

        # kf_model
        kf_cascades = [
            None for _ in range((num_cascades - kf_num_cascades) if kf_no_parameter_sharing else 0)
        ]
        if kf_model == "KSpaceModule3D":
            kf_cascades.extend([
                KSpaceSequentialModule(
                    nn.ModuleList([
                        KSpaceModule3D(
                            model=nn.Sequential(*[
                                KSpaceKernel3D(
                                    in_channels=kf_inp_channels * 2 if kf_conv_mode != 'complex' else kf_inp_channels,
                                    out_channels=kf_out_channels * 2 if kf_conv_mode != 'complex' else kf_out_channels,
                                    conv_mode=kf_conv_mode,
                                    kernel_size=kf_kernel_sizes,
                                    padding=kf_paddings,
                                    acti_func=kf_acti_func,
                                    bias=False,
                                ) for i in range(kf_num_blocks)
                            ]),
                            conv_mode=kf_conv_mode,
                            physics=True,
                            normalized=kf_normalize,
                        )
                    ])
                ) for _ in range(kf_num_cascades if kf_no_parameter_sharing else 1)
            ])
        elif kf_model == "none":
            kf_cascades.extend([None for _ in range(kf_num_cascades if kf_no_parameter_sharing else 1)])
        else:
            raise
        self.kf_cascades = nn.ModuleList(kf_cascades)

        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._temporal_dim = 2
        self._spatial_dims = (3, 4)

        self.num_cascades = num_cascades
        self.xt_no_parameter_sharing = xt_no_parameter_sharing
        self.kt_no_parameter_sharing = kt_no_parameter_sharing
        self.xf_no_parameter_sharing = xf_no_parameter_sharing
        self.kf_no_parameter_sharing = kf_no_parameter_sharing
        
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        additional_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            masked_kspace: b, c, d, h, w, comp
            mask: b, 1, d, h, w, 1
            additional_masks: b, 3, s, 1, d, h, w, 1
        """
        # sensitivity_map
        sens_maps = self.sens_net(
            masked_kspace=masked_kspace,
            mask=additional_masks[:,0,0,...], # center_mask
        )

        kspace_pred = masked_kspace.clone()

        if self.kf_cascades[-1] is not None:
            kf_masked_kspace = self.forward_operator(masked_kspace, dim=[self._temporal_dim])

        kt_acs, kf_acs = [], []
        zero = torch.zeros_like(masked_kspace)
        for step in range(self.num_cascades):
            ### xt_model
            xt_cascade = self.xt_cascades[step] if self.xt_no_parameter_sharing else self.xt_cascades[0]
            if xt_cascade is not None:
                kspace_pred = xt_cascade(
                    current_kspace=kspace_pred,
                    ref_kspace=masked_kspace,
                    mask=mask,
                    sens_maps=sens_maps
                )

            ### kt_model
            kt_cascade = self.kt_cascades[step] if self.kt_no_parameter_sharing else self.kt_cascades[0]
            if kt_cascade is not None:
                kt_pred = kt_cascade(
                    x=kspace_pred,
                    mask=additional_masks[:,2,...], # shrinked_region_mask
                )
                kspace_pred = torch.where(mask, kspace_pred, kt_pred)

                # Auto-Calibration Signal (ACS)
                acs = kt_cascade(
                    x=masked_kspace,
                    mask=additional_masks[:,1,...], # shrinked_center_mask
                )
                kt_acs.append(acs)

            ### xf_model
            xf_cascade = self.xf_cascades[step] if self.xf_no_parameter_sharing else self.xf_cascades[0]
            if xf_cascade is not None:
                kspace_pred = xf_cascade(
                    current_kspace=kspace_pred,
                    ref_kspace=masked_kspace,
                    mask=mask,
                    sens_maps=sens_maps
                )

            ### kf_model
            kf_cascade = self.kf_cascades[step] if self.kf_no_parameter_sharing else self.kf_cascades[0]
            if kf_cascade is not None:
                kspace_pred = self.forward_operator(kspace_pred, dim=[self._temporal_dim])
                kf_pred = kf_cascade(
                    x=kspace_pred,
                    mask=additional_masks[:,2,...], # shrinked_region_mask
                )
                kspace_pred = torch.where(mask, kspace_pred, kf_pred)
                kspace_pred = self.backward_operator(kspace_pred, dim=[self._temporal_dim])

                # Auto-Calibration Signal (ACS)
                acs = kf_cascade(
                    x=kf_masked_kspace,
                    mask=additional_masks[:,1,...], # shrinked_center_mask
                )
                kf_acs.append(acs)

        # kt_num_cascades, b, c, h, w, comp
        if len(kt_acs) == 0:
            kt_acs = None
        else:
            kt_acs = torch.stack(kt_acs, dim=0)

        # kf_num_cascades, b, c, h, w, comp
        if len(kf_acs) == 0:
            kf_acs = None
        else:
            kf_acs = torch.stack(kf_acs, dim=0)

        kspace_preds = kspace_pred[None]

        return kspace_preds, kt_acs, kf_acs, sens_maps
