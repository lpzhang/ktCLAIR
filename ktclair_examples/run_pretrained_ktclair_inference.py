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

import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri
from fastmri.data.subsample import create_mask_for_mask_type
import numpy as np
import requests
import torch
from ktclair.data.cmr_data import CMRDataset
from ktclair.data.transforms import CMRDataTransform
from ktclair.models.ktclair import KTCLAIR
from tqdm import tqdm

from typing import Dict, Tuple, Union
from ktclair.utils import str_to_class

import h5py
import hdf5storage
from einops import rearrange

KTCLAIR_FOLDER = "CLAIR"
MODEL_FNAMES = {
    "cine": "ktclair_test_examples/checkpoints/cine.pt",
    "mapping": "ktclair_test_examples/checkpoints/mapping.pt",
}


def load_model(task):
    if task.lower() == 'cine':
        model = KTCLAIR(
            forward_operator = str_to_class("ktclair", "fft2"),
            backward_operator = str_to_class("ktclair", "ifft2"),
            ### sens_model
            sens_model="Sensitivity3DModule",
            sens_chans=8,
            sens_pools=4,
            mask_center=True,
            ### xt_model
            xt_model="NormUnet3DXF",
            xt_num_cascades=12,
            xt_inp_channels=2,
            xt_out_channels=2,
            xt_chans=32,
            xt_pools=4,
            xt_dc_mode='GD',
            xt_no_parameter_sharing=True,
            ### kt_model
            kt_model="KSpaceModule3D",
            kt_num_cascades=12,
            kt_num_blocks=4,
            kt_inp_channels=10,
            kt_out_channels=10,
            kt_conv_mode='complex', # 'complex'/'real'
            kt_kernel_sizes=(3, 3, 3),
            kt_paddings=(1, 1, 1),
            kt_acti_func="none", # "ReLU"/"ReLU"/"none"
            kt_no_parameter_sharing=True,
            kt_normalize=True,
            ### xf_model
            xf_model="NormUnet3D",
            xf_num_cascades=12,
            xf_inp_channels=2,
            xf_out_channels=2,
            xf_chans=32,
            xf_pools=4,
            xf_dc_mode='GD',
            xf_no_parameter_sharing=True,
            ### kf_model
            kf_model="KSpaceModule3D",
            kf_num_cascades=12,
            kf_num_blocks=4,
            kf_inp_channels=10,
            kf_out_channels=10,
            kf_conv_mode='complex', # 'complex'/'real'
            kf_kernel_sizes=(3, 3, 3),
            kf_paddings=(1, 1, 1),
            kf_acti_func="none", # "ReLU"/"ReLU"/"none"
            kf_no_parameter_sharing=True,
            kf_normalize=True,
        )
    elif task.lower() == 'mapping':
        model = KTCLAIR(
            forward_operator = str_to_class("ktclair", "fft2"),
            backward_operator = str_to_class("ktclair", "ifft2"),
            ### sens_model
            sens_model="Sensitivity3DModule",
            sens_chans=8,
            sens_pools=4,
            mask_center=True,
            ### xt_model
            xt_model="NormUnet3DXF",
            xt_num_cascades=12,
            xt_inp_channels=2,
            xt_out_channels=2,
            xt_chans=64,
            xt_pools=4,
            xt_dc_mode='GD',
            xt_no_parameter_sharing=True,
            ### kt_model
            kt_model="KSpaceModule3D",
            kt_num_cascades=12,
            kt_num_blocks=4,
            kt_inp_channels=10,
            kt_out_channels=10,
            kt_conv_mode='complex', # 'complex'/'real'
            kt_kernel_sizes=(3, 3, 3),
            kt_paddings=(1, 1, 1),
            kt_acti_func="none", # "ReLU"/"ReLU"/"none"
            kt_no_parameter_sharing=True,
            kt_normalize=True,
            ### xf_model
            xf_model="NormUnet3D",
            xf_num_cascades=12,
            xf_inp_channels=2,
            xf_out_channels=2,
            xf_chans=64,
            xf_pools=4,
            xf_dc_mode='GD',
            xf_no_parameter_sharing=True,
            ### kf_model
            kf_model="KSpaceModule3D",
            kf_num_cascades=12,
            kf_num_blocks=4,
            kf_inp_channels=10,
            kf_out_channels=10,
            kf_conv_mode='complex', # 'complex'/'real'
            kf_kernel_sizes=(3, 3, 3),
            kf_paddings=(1, 1, 1),
            kf_acti_func="none", # "ReLU"/"ReLU"/"none"
            kf_no_parameter_sharing=True,
            kf_normalize=True,
        )
    else:
        raise

    return model


def load_data_transform(task):
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        mask_type_str="equispaced",
        center_fractions=(0.08, 0.08, 0.08),
        accelerations=(4, 8, 10),
    )

    if task.lower() == 'cine':
        data_transform = CMRDataTransform(
            mask_func=mask,
            spatial_shrink_size=4,
            scontext_shrink_size=0,
            tfcontext_shrink_size=1,
            mask_correction_mode=None,
        )
    elif task.lower() == 'mapping':
        data_transform = CMRDataTransform(
            mask_func=mask,
            spatial_shrink_size=4,
            scontext_shrink_size=0,
            tfcontext_shrink_size=1,
            mask_correction_mode=('sy'),
        )
    else:
        raise

    return data_transform


def run_ktclair_model(batch, model, device):
    kspace_pred, _, _, _ = model(
        batch.masked_kspace.to(device), batch.mask.to(device), batch.additional_masks.to(device)
    ) # (num_cascades b c d h w comp)
    kspace_pred = kspace_pred.cpu()

    target_slice = kspace_pred.shape[-4]//2
    kspace_pred = kspace_pred[-1,:,:,target_slice,:,:,:] # (b c h w comp)
    output = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1) # (b h w)

    # scaling back
    output = output * batch.sfactor.view(-1, *((1,) * (len(output.shape) - 1)))

    return output[0], int(batch.tframe_num[0]), int(batch.slice_num[0]), batch.fname[0]


def save_reconstructions(reconstructions: Dict[str, Dict[str, np.ndarray]], out_dir: Path, recon_key: str = "img4ranking"):
    for fname, recons in reconstructions.items():
        out_fpath = out_dir / fname
        out_fpath.parent.mkdir(exist_ok=True, parents=True)
        # with h5py.File(out_fpath, "w") as hf:
        #     hf.create_dataset(recon_key, data=recons)
        #
        # convert to have the same dimension order as the original data when using hdf5storage.savemat
        recons = rearrange(recons, 'tf sz sy sx -> sx sy sz tf')
        hdf5storage.savemat(str(out_fpath), {recon_key:recons})


def run_inference(
    data_path: Union[Path, str],
    output_path: Union[Path, str],
    task: str,
    state_dict_file: Union[Path, str] = None,
    challenge: str = "MultiCoil",
    setname: str = "TestSet",
    accfactor: Tuple[str] = ("AccFactor04", "AccFactor08", "AccFactor10"),
    acquisitions: Tuple[str] = ("cine_lax", "cine_sax", "T1map", "T2map"),
    device: str = "cuda",
):
    model = load_model(task)
    # raise if we don't have the state_dict
    if state_dict_file is None:
        state_dict_file = Path(KTCLAIR_FOLDER) / MODEL_FNAMES[task.lower()]
    assert Path(state_dict_file).exists(), "we don't have the state_dict"

    model.load_state_dict(torch.load(state_dict_file))

    model = model.eval()

    # data loader setup
    data_transform = load_data_transform(task)
    dataset = CMRDataset(
        # root=data_path,
        root=Path(data_path) / challenge / task / setname,
        task=task,
        transform=data_transform,
        challenge=challenge,
        accfactor=accfactor,
        acquisitions=acquisitions,
        data_partition="test",
        scontext=0,
        tfcontext=1
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=16)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(dict)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, tframe_num, slice_num, fname = run_ktclair_model(batch, model, device)

        outputs[fname][f"t{tframe_num}-s{slice_num}"] = output # (h w)

    # stack all the slices for each file
    for fname in outputs:
        num_tframes = len(set([k.split('-')[0] for k in outputs[fname].keys()]))
        num_slices  = len(set([k.split('-')[1] for k in outputs[fname].keys()]))
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname].items(),
            key=lambda k: (int(k[0].split('-')[0].split('t')[1]), int(k[0].split('-')[1].split('s')[1]))
        )]) # (tf*sz h w)
        outputs[fname] = outputs[fname].reshape((num_tframes, num_slices) + outputs[fname].shape[1:]
        ) # (tf sz h w)

    # save outputs
    save_reconstructions(outputs, Path(output_path))

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )
    parser.add_argument(
        "--task",
        choices=("Cine","Mapping"),
        type=str,
        required=True,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict",
    )
    parser.add_argument(
        "--challenge",
        default="MultiCoil",
        choices=(
            "MultiCoil",
            "SingleCoil",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--setname",
        default="TestSet",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--accfactor",
        nargs="+",
        default=("AccFactor04", "AccFactor08", "AccFactor10"),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--acquisitions",
        nargs="+",
        default=("cine_lax", "cine_sax", "T1map", "T2map"),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )

    args = parser.parse_args()

    run_inference(
        args.data_path,
        args.output_path,
        args.task,
        args.state_dict_file,
        args.challenge,
        args.setname,
        args.accfactor,
        args.acquisitions,
        torch.device(args.device),
    )
