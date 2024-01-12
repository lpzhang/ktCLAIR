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

from argparse import ArgumentParser
from typing import Optional, Tuple, Union, Sequence

import fastmri
import torch
from fastmri.data import transforms

from ktclair.pl_modules.mri_module import MriModule

from ktclair.losses import SSIMLoss

from ktclair.utils import str_to_class

from ktclair.models.ktclair import KTCLAIR

import distutils.util


class KTCLAIRModule(MriModule):
    """
    KTCLAIR training module.
    """

    def __init__(
        self,
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
        ### loss setttings
        loss_num_cascades: int = 1,
        loss_num_slices: int = 1,
        loss_names: Sequence[str] = ["ssim", "l1", "xt", "roi", "kt", "acs"],
        loss_weights: Tuple[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_decay: bool = False,
        loss_multiscale: Tuple[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        use_scan_stats: bool = True,
        crop_target: bool = False,
        ### optimizer
        optimizer: str = 'Adam',
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.forward_operator = str_to_class("ktclair", "fft2")
        self.backward_operator = str_to_class("ktclair", "ifft2")
        ### sens_model
        self.sens_model=sens_model
        self.sens_chans=sens_chans
        self.sens_pools=sens_pools
        self.mask_center=mask_center
        ### xt_model
        self.xt_model=xt_model
        self.xt_num_cascades=xt_num_cascades
        self.xt_inp_channels=xt_inp_channels
        self.xt_out_channels=xt_out_channels
        self.xt_chans=xt_chans
        self.xt_pools=xt_pools
        self.xt_dc_mode=xt_dc_mode
        self.xt_no_parameter_sharing=xt_no_parameter_sharing
        self.xt_bg_drop_prob=xt_bg_drop_prob
        ### kt_model
        self.kt_model=kt_model
        self.kt_num_cascades=kt_num_cascades
        self.kt_num_blocks=kt_num_blocks
        self.kt_inp_channels=kt_inp_channels
        self.kt_out_channels=kt_out_channels
        self.kt_conv_mode=kt_conv_mode
        self.kt_kernel_sizes=kt_kernel_sizes
        self.kt_paddings=kt_paddings
        self.kt_acti_func=kt_acti_func
        self.kt_no_parameter_sharing=kt_no_parameter_sharing
        self.kt_normalize=kt_normalize
        ### xf_model
        self.xf_model=xf_model
        self.xf_num_cascades=xf_num_cascades
        self.xf_inp_channels=xf_inp_channels
        self.xf_out_channels=xf_out_channels
        self.xf_chans=xf_chans
        self.xf_pools=xf_pools
        self.xf_dc_mode=xf_dc_mode
        self.xf_no_parameter_sharing=xf_no_parameter_sharing
        self.xf_bg_drop_prob=xf_bg_drop_prob
        ### kf_model
        self.kf_model=kf_model
        self.kf_num_cascades=kf_num_cascades
        self.kf_num_blocks=kf_num_blocks
        self.kf_inp_channels=kf_inp_channels
        self.kf_out_channels=kf_out_channels
        self.kf_conv_mode=kf_conv_mode
        self.kf_kernel_sizes=kf_kernel_sizes
        self.kf_paddings=kf_paddings
        self.kf_acti_func=kf_acti_func
        self.kf_no_parameter_sharing=kf_no_parameter_sharing
        self.kf_normalize=kf_normalize

        self.model = KTCLAIR(
            forward_operator=self.forward_operator,
            backward_operator=self.backward_operator,
            ### sens_model
            sens_model=self.sens_model,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            mask_center=self.mask_center,
            ### xt_model
            xt_model=self.xt_model,
            xt_num_cascades=self.xt_num_cascades,
            xt_inp_channels=self.xt_inp_channels,
            xt_out_channels=self.xt_out_channels,
            xt_chans=self.xt_chans,
            xt_pools=self.xt_pools,
            xt_dc_mode=self.xt_dc_mode,
            xt_no_parameter_sharing=self.xt_no_parameter_sharing,
            xt_bg_drop_prob=self.xt_bg_drop_prob,
            ### kt_model
            kt_model=self.kt_model,
            kt_num_cascades=self.kt_num_cascades,
            kt_num_blocks=self.kt_num_blocks,
            kt_inp_channels=self.kt_inp_channels,
            kt_out_channels=self.kt_out_channels,
            kt_conv_mode=self.kt_conv_mode,
            kt_kernel_sizes=self.kt_kernel_sizes,
            kt_paddings=self.kt_paddings,
            kt_acti_func=self.kt_acti_func,
            kt_no_parameter_sharing=self.kt_no_parameter_sharing,
            kt_normalize=self.kt_normalize,
            ### xf_model
            xf_model=self.xf_model,
            xf_num_cascades=self.xf_num_cascades,
            xf_inp_channels=self.xf_inp_channels,
            xf_out_channels=self.xf_out_channels,
            xf_chans=self.xf_chans,
            xf_pools=self.xf_pools,
            xf_dc_mode=self.xf_dc_mode,
            xf_no_parameter_sharing=self.xf_no_parameter_sharing,
            xf_bg_drop_prob=self.xf_bg_drop_prob,
            ### kf_model
            kf_model=self.kf_model,
            kf_num_cascades=self.kf_num_cascades,
            kf_num_blocks=self.kf_num_blocks,
            kf_inp_channels=self.kf_inp_channels,
            kf_out_channels=self.kf_out_channels,
            kf_conv_mode=self.kf_conv_mode,
            kf_kernel_sizes=self.kf_kernel_sizes,
            kf_paddings=self.kf_paddings,
            kf_acti_func=self.kf_acti_func,
            kf_no_parameter_sharing=self.kf_no_parameter_sharing,
            kf_normalize=self.kf_normalize,
        )

        # loss settings
        self.loss_num_cascades=loss_num_cascades
        self.loss_num_slices=loss_num_slices
        assert len(loss_names) == len(loss_weights)
        self.loss_names=loss_names
        self.loss_weights = {name:weight for (name, weight) in zip(loss_names, loss_weights)}
        self.loss_decay=loss_decay
        self.loss_multiscale=loss_multiscale
        self.use_scan_stats=use_scan_stats
        self.crop_target=crop_target
        # optimizer
        self.optimizer=optimizer
        self.lr=lr
        self.lr_step_size=lr_step_size
        self.lr_gamma=lr_gamma
        self.weight_decay=weight_decay
        self.momentum=momentum

        self.loss = SSIMLoss()
        self._coil_dim = 1
        self._temporal_dim = 2
        self._spatial_dims = (3, 4)

        # decay factor
        num_cascades = 0
        for name, num in zip(
            (xt_model, kt_model, xf_model, kf_model),
            (xt_num_cascades, kt_num_cascades, xf_num_cascades, kf_num_cascades)
        ):
            num_cascades = num_cascades if name.lower() == 'none' else max(num_cascades, num)
        self.decays = [(2**(i+1)) if self.loss_decay else 1.0 for i in range(-num_cascades,0)]

    def forward(self, masked_kspace, mask, additional_masks):
        return self.model(masked_kspace, mask, additional_masks)

    def one_step_xt_loss(self, preds, gt, data_range):
        """ preds (loss_num_cascades b d h w)
            gt (b d h w)
        """
        loss = 0
        for idx in range(-self.loss_num_cascades, 0):
            cascade_loss = 0
            if "ssim" in self.loss_names:
                cascade_loss += self.loss_weights['ssim'] * self.loss(
                    preds[idx], gt, data_range=data_range
                )
            if "l1" in self.loss_names:
                cascade_loss += self.loss_weights['l1'] * torch.nn.functional.l1_loss(
                    preds[idx], gt
                )
            loss += self.decays[idx] * cascade_loss
        return loss

    def one_step_ms_xt_loss(self, preds, gt, data_range, crop_size):
        """ preds (loss_num_cascades b d h w)
            gt (b d h w)
        """
        # multiscale xt_loss
        loss = 0
        for scale in self.loss_multiscale:
            new_crop_size = (
                torch.round(crop_size[0] * scale).to(crop_size[0]),
                torch.round(crop_size[1] * scale).to(crop_size[1])
            )
            crop_preds = transforms.center_crop(preds, new_crop_size)
            crop_gt = transforms.center_crop(gt, new_crop_size)
            loss += self.one_step_xt_loss(crop_preds, crop_gt, data_range)
        return loss

    def one_step_kt_loss(self, preds, gt):
        """ preds (loss_num_cascades b c d h w comp)
            gt (b c d h w comp)
        """
        loss = 0
        for idx in range(-self.loss_num_cascades, 0):
            loss += self.decays[idx] * torch.nn.functional.l1_loss(
                preds[idx], gt
            )
        return loss

    def one_step_acs_loss(self, preds, gt):
        """ preds (kt_num_cascades b c d h w comp)
            gt (b c d h w comp)
        """
        loss = 0
        for idx in range(-preds.shape[0], 0):
            loss += self.decays[idx] * torch.nn.functional.l1_loss(
                preds[idx], gt
            )
        return loss

    def training_step(self, batch, batch_idx):
        """ 6D batch data with shape (b c d h w comp)
            batch.kspace (b sc tf/sz sy sx 2)
            batch.masked_kspace (b sc tf/sz sy sx 2)
            batch.mask (b 1 1 sy sx 1)
            batch.additional_masks (3 s 1 tf/sz sy sx 1)
            batch.target (b tf/sz sy sx)
        """
        ### forward
        kspace_preds, kt_acs, kf_acs, _ = self.forward(
            batch.masked_kspace, batch.mask, batch.additional_masks,
        ) # (num_cascades b c d h w comp), (kt_num_cascades b c d h w comp), (kf_num_cascades b c d h w comp), (b c d h w comp)

        ### select slices for loss calculation
        if self.loss_num_slices > 0:
            num_slices = kspace_preds.shape[-4]
            slice_start = max((num_slices - self.loss_num_slices) // 2, 0) # favor left when even
            slice_end = min(slice_start + self.loss_num_slices, num_slices)
            loss_slice = slice(slice_start, slice_end)
        else:
            loss_slice = slice(None)
        kspace_preds = kspace_preds[...,loss_slice,:,:,:]
        kt_acs = kt_acs[...,loss_slice,:,:,:] if kt_acs is not None else None
        kf_acs = kf_acs[...,loss_slice,:,:,:] if kf_acs is not None else None
        kspace = batch.kspace[...,loss_slice,:,:,:]
        target = batch.target[...,loss_slice,:,:]
        masked_kspace = batch.masked_kspace[...,loss_slice,:,:,:]
        roi = batch.seg[...,loss_slice,:,:].to(target)
        
        ### select cascades for loss calculation
        kspace_preds = kspace_preds[-self.loss_num_cascades:] # loss_num_cascades b c d h w comp
        image_preds = self.backward_operator(
            kspace_preds, dim=(self._spatial_dims[0] + 1, self._spatial_dims[1] + 1)
        ) # loss_num_cascades b c d h w comp
        outputs = fastmri.rss(
            fastmri.complex_abs(image_preds), dim=self._coil_dim + 1
        ) # loss_num_cascades b d h w

        ### loss calculation
        if "ssim" in self.loss_names:
            if self.use_scan_stats:
                # using the volume maximum value for fastmri dataset
                data_range = batch.max_value / batch.sfactor
            else:
                # using the batch maximum value of the target
                data_range = target.max().expand(target.shape[0]).to(target)
        else:
            data_range = None
        # xt_loss
        xt_loss = 0
        if 'xt' in self.loss_names:
            xt_loss += self.loss_weights['xt'] * self.one_step_ms_xt_loss(
                outputs, target, data_range, batch.crop_size,
            )
        # roi_xt_loss
        roi_xt_loss = 0
        if 'roi' in self.loss_names:
            roi_target = target * roi
            roi_outputs = outputs * roi[None]
            roi_xt_loss += self.loss_weights['roi'] * self.one_step_xt_loss(
                roi_outputs, roi_target, data_range,
            )
        # kt_loss
        kt_loss = 0
        if 'kt' in self.loss_names:
            kt_loss += self.loss_weights['kt'] * self.one_step_kt_loss(
                kspace_preds, kspace,
            )
        # acs_loss
        acs_loss = 0
        if 'acs' in self.loss_names:
            if kt_acs is not None:
                acs_loss += self.one_step_acs_loss(
                    kt_acs, masked_kspace,
                )
            if kf_acs is not None:
                acs_loss += self.one_step_acs_loss(
                    kf_acs, self.forward_operator(masked_kspace, dim=[self._temporal_dim]),
                )
            acs_loss *= self.loss_weights['acs']
        # total loss
        loss = xt_loss + roi_xt_loss + kt_loss + acs_loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        kspace_preds, kt_acs, kf_acs, _ = self.forward(
            batch.masked_kspace, batch.mask, batch.additional_masks,
        ) # (num_cascades b c d h w comp)

        ### select slices for loss calculation
        if self.loss_num_slices > 0:
            num_slices = kspace_preds.shape[-4]
            slice_start = max((num_slices - self.loss_num_slices) // 2, 0) # favor left when even
            slice_end = min(slice_start + self.loss_num_slices, num_slices)
            loss_slice = slice(slice_start, slice_end)
        else:
            loss_slice = slice(None)
        kspace_preds = kspace_preds[...,loss_slice,:,:,:]
        kt_acs = kt_acs[...,loss_slice,:,:,:] if kt_acs is not None else None
        kf_acs = kf_acs[...,loss_slice,:,:,:] if kf_acs is not None else None
        kspace = batch.kspace[...,loss_slice,:,:,:]
        target = batch.target[...,loss_slice,:,:]
        masked_kspace = batch.masked_kspace[...,loss_slice,:,:,:]
        roi = batch.seg[...,loss_slice,:,:].to(target)

        ### select cascades for loss calculation
        kspace_preds = kspace_preds[-self.loss_num_cascades:] # loss_num_cascades b c d h w comp
        image_preds = self.backward_operator(
            kspace_preds, dim=(self._spatial_dims[0] + 1, self._spatial_dims[1] + 1)
        ) # loss_num_cascades b c d h w comp
        outputs = fastmri.rss(
            fastmri.complex_abs(image_preds), dim=self._coil_dim + 1
        ) # loss_num_cascades b d h w

        ### loss calculation
        if "ssim" in self.loss_names:
            if self.use_scan_stats:
                # using the volume maximum value for fastmri dataset
                data_range = batch.max_value / batch.sfactor
            else:
                # using the batch maximum value of the target
                data_range = target.max().expand(target.shape[0]).to(target)
        else:
            data_range = None
        # xt_loss
        xt_loss = 0
        if 'xt' in self.loss_names:
            xt_loss += self.loss_weights['xt'] * self.one_step_ms_xt_loss(
                outputs, target, data_range, batch.crop_size,
            )
        # roi_xt_loss
        roi_xt_loss = 0
        if 'roi' in self.loss_names:
            roi_target = target * roi
            roi_outputs = outputs * roi[None]
            roi_xt_loss += self.loss_weights['roi'] * self.one_step_xt_loss(
                roi_outputs, roi_target, data_range,
            )
        # kt_loss
        kt_loss = 0
        if 'kt' in self.loss_names:
            kt_loss += self.loss_weights['kt'] * self.one_step_kt_loss(
                kspace_preds, kspace,
            )
        # acs_loss
        acs_loss = 0
        if 'acs' in self.loss_names:
            if kt_acs is not None:
                acs_loss += self.one_step_acs_loss(
                    kt_acs, masked_kspace,
                )
            if kf_acs is not None:
                acs_loss += self.one_step_acs_loss(
                    kf_acs, self.forward_operator(masked_kspace, dim=[self._temporal_dim]),
                )
            acs_loss *= self.loss_weights['acs']
        # total loss
        loss = xt_loss + roi_xt_loss + kt_loss + acs_loss

        ### output
        # scaling back
        target = transforms.center_crop(target, batch.crop_size)
        target, output = transforms.center_crop_to_smallest(target, outputs[-1])
        target = target * batch.sfactor.view(-1, *((1,) * (len(target.shape) - 1)))
        output = output * batch.sfactor.view(-1, *((1,) * (len(output.shape) - 1)))

        # target slice
        target_slice = target.shape[1]//2

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "tframe_num": batch.tframe_num,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output[:,target_slice,...],
            "target": target[:,target_slice,...],
            "val_loss": loss,
        }

    # def test_step(self, batch, batch_idx):
    #     kspace_preds, _, _ = self.forward(
    #         batch.masked_kspace, batch.mask, batch.additional_masks,
    #     )
    #     # b, c, d, h, w, comp
    #     output = kspace_preds[-1,...]
    #     output = fastmri.rss(
    #         fastmri.complex_abs(self.backward_operator(output, dim=self._spatial_dims)),
    #         dim=self._coil_dim
    #     )

    #     # check for FLAIR 203
    #     if output.shape[-1] < batch.crop_size[1]:
    #         crop_size = (output.shape[-1], output.shape[-1])
    #     else:
    #         crop_size = batch.crop_size

    #     output = transforms.center_crop(output, crop_size)

    #     # scaling back
    #     output = output * batch.sfactor.view(-1, *((1,) * (len(output.shape) - 1)))

    #     # target slice
    #     target_slice = target.shape[1]//2

    #     return {
    #         "fname": batch.fname,
    #         "slice": batch.slice_num,
    #         "output": output[:,target_slice,...].cpu().numpy(),
    #     }

    def test_step(self, batch, batch_idx):
        kspace_preds, _, _, _ = self.forward(
            batch.masked_kspace, batch.mask, batch.additional_masks,
        ) # (num_cascades b c d h w comp)

        output = kspace_preds[-1] # b, c, d, h, w, comp
        output = fastmri.rss(
            fastmri.complex_abs(self.backward_operator(output, dim=self._spatial_dims)),
            dim=self._coil_dim
        )

        if self.crop_target:
            target = transforms.center_crop(batch.target, batch.crop_size)
        else:
            target = batch.target

        target, output = transforms.center_crop_to_smallest(target, output)

        # scaling back
        target = target * batch.sfactor.view(-1, *((1,) * (len(target.shape) - 1)))
        output = output * batch.sfactor.view(-1, *((1,) * (len(output.shape) - 1)))

        # target slice
        target_slice = target.shape[1]//2

        return {
            "fname": batch.fname,
            "tframe_num": batch.tframe_num,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "recons_key": batch.recons_key,
            "output": output[:,target_slice,...],
            "target": target[:,target_slice,...],
        }

    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                momentum=self.momentum, nesterov=True,
            )
        elif self.optimizer.lower() == 'adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adamw':
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            assert False and "Invalid optimizer"

        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params

        ### sens_model
        parser.add_argument(
            "--sens_model",
            choices=("Sensitivity3DModule", ""),
            default="Sensitivity3DModule",
            type=str,
            help="Which Sensitivity Model to use",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=int,
            help="Number of channels for sense map estimation U-Net",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net",
        )
        parser.add_argument(
            "--mask_center",
            default=True,
            type=bool,
            help="Whether to mask center of k-space for sensitivity map calculation",
        )
        ### xt_model
        parser.add_argument(
            "--xt_model",
            default="NormUnet3D",
            type=str,
            help="Which xt model to use",
        )
        parser.add_argument(
            "--xt_num_cascades",
            default=12,
            type=int,
            help="Number of xt cascades",
        )
        parser.add_argument(
            "--xt_inp_channels",
            default=2,
            type=int,
            help="Number of xt input channels",
        )
        parser.add_argument(
            "--xt_out_channels",
            default=2,
            type=int,
            help="Number of xt output chanenls",
        )
        parser.add_argument(
            "--xt_chans",
            default=48,
            type=int,
            help="Number of xt feature chanenls",
        )
        parser.add_argument(
            "--xt_pools",
            default=4,
            type=int,
            help="Number of xt pooling layers",
        )
        parser.add_argument(
            "--xt_dc_mode",
            default="GD",
            type=str,
            help="",
        )
        parser.add_argument(
            "--xt_no_parameter_sharing",
            default=True,
            type=bool,
            help="",
        )
        parser.add_argument(
            "--xt_bg_drop_prob",
            default=None,
            type=float,
            help="",
        )
        ### kt_model
        parser.add_argument(
            "--kt_model",
            default="KSpaceModule3D",
            type=str,
            help="Which kt model to use",
        )
        parser.add_argument(
            "--kt_num_cascades",
            default=12,
            type=int,
            help="Number of kt cascades"
        )
        parser.add_argument(
            "--kt_num_blocks",
            default=4,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kt_inp_channels",
            default=10,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kt_out_channels",
            default=10,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kt_conv_mode",
            choices=("real", "complex"),
            default="complex",
            type=str,
            help="",
        )
        parser.add_argument(
            "--kt_kernel_sizes",
            nargs="+",
            default=[3, 3, 3],
            type=int,
            help="",
        )
        parser.add_argument(
            "--kt_paddings",
            nargs="+",
            default=[1, 1, 1],
            type=int,
            help="",
        )
        parser.add_argument(
            "--kt_acti_func",
            nargs="+",
            default="none",
            type=str,
            help="ReLU, LeakyReLU, none",
        )
        parser.add_argument(
            "--kt_no_parameter_sharing",
            default=True,
            type=bool,
            help="",
        )
        parser.add_argument(
            "--kt_normalize",
            default=True,
            type=lambda x:bool(distutils.util.strtobool(x)),
            help="",
        )
        ### xf_model
        parser.add_argument(
            "--xf_model",
            default="NormUnet3D",
            type=str,
            help="Which xf model to use",
        )
        parser.add_argument(
            "--xf_num_cascades",
            default=12,
            type=int,
            help="Number of xf cascades",
        )
        parser.add_argument(
            "--xf_inp_channels",
            default=2,
            type=int,
            help="Number of xf input channels",
        )
        parser.add_argument(
            "--xf_out_channels",
            default=2,
            type=int,
            help="Number of xf output chanenls",
        )
        parser.add_argument(
            "--xf_chans",
            default=48,
            type=int,
            help="Number of xf feature chanenls",
        )
        parser.add_argument(
            "--xf_pools",
            default=4,
            type=int,
            help="Number of xf pooling layers",
        )
        parser.add_argument(
            "--xf_dc_mode",
            default="GD",
            type=str,
            help="",
        )
        parser.add_argument(
            "--xf_no_parameter_sharing",
            default=True,
            type=bool,
            help="",
        )
        parser.add_argument(
            "--xf_bg_drop_prob",
            default=None,
            type=float,
            help="",
        )
        ### kf_model
        parser.add_argument(
            "--kf_model",
            default="KSpaceModule3D",
            type=str,
            help="Which kf model to use",
        )
        parser.add_argument(
            "--kf_num_cascades",
            default=12,
            type=int,
            help="Number of kf cascades"
        )
        parser.add_argument(
            "--kf_num_blocks",
            default=4,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kf_inp_channels",
            default=10,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kf_out_channels",
            default=10,
            type=int,
            help="",
        )
        parser.add_argument(
            "--kf_conv_mode",
            choices=("real", "complex"),
            default="complex",
            type=str,
            help="",
        )
        parser.add_argument(
            "--kf_kernel_sizes",
            nargs="+",
            default=[3, 3, 3],
            type=int,
            help="",
        )
        parser.add_argument(
            "--kf_paddings",
            nargs="+",
            default=[1, 1, 1],
            type=int,
            help="",
        )
        parser.add_argument(
            "--kf_acti_func",
            nargs="+",
            default="none",
            type=str,
            help="ReLU, LeakyReLU, none",
        )
        parser.add_argument(
            "--kf_no_parameter_sharing",
            default=True,
            type=bool,
            help="",
        )
        parser.add_argument(
            "--kf_normalize",
            default=True,
            type=lambda x:bool(distutils.util.strtobool(x)),
            help="",
        )

        ### loss settings
        parser.add_argument(
            "--loss_num_cascades",
            default=1,
            type=int,
            help="",
        )
        parser.add_argument(
            "--loss_num_slices",
            default=1,
            type=int,
            help="",
        )
        parser.add_argument(
            "--loss_names",
            nargs="+",
            default=["ssim", "l1", "xt", "roi", "kt", "acs"],
            type=str,
            help="",
        )
        parser.add_argument(
            "--loss_weights",
            nargs="+",
            default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            type=float,
            help="",
        )
        parser.add_argument(
            "--loss_decay",
            default=False,
            type=lambda x:bool(distutils.util.strtobool(x)),
            help="",
        )
        parser.add_argument(
            "--loss_multiscale",
            nargs="+",
            default=[0.5, 0.75, 1.0, 1.25, 1.5],
            type=float,
            help="",
        )
        parser.add_argument(
            "--use_scan_stats",
            default=True,
            type=bool,
            help="",
        )
        parser.add_argument(
            "--crop_target",
            default=False,
            type=lambda x:bool(distutils.util.strtobool(x)),
            help="",
        )

        # training params (opt)
        parser.add_argument(
            "--optimizer", default='Adam', type=str, help="optimization algorithm"
        )
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--momentum",
            default=0.99,
            type=float,
            help="SGD momentum factor",
        )

        return parser
