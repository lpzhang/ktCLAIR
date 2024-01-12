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

import logging
import os
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import yaml
import pandas as pd


class CMRDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        task: str,
        accfactor: Sequence[str] = None,
        transform: Optional[Callable] = None,
        acquisitions: Optional[Sequence[str]] = None,
        scontext: Optional[int] = None,
        tfcontext: Optional[int] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            acquisitions: Optional; If provided, only slices with the desired
                MRI pulse sequence will be considered.
        """
        if challenge not in ("SingleCoil", "MultiCoil"):
            raise ValueError('challenge should be either "SingleCoil" or "MultiCoil"')
        self.challenge = challenge

        if task not in ("Cine", "Mapping"):
            raise ValueError('task should be either "Cine" or "Mapping"')
        self.task = task

        self.transform = transform

        self.examples = []

        self.scontext = scontext
        self.tfcontext = tfcontext

        # get accfactor
        if accfactor is not None:
            accfactor = [accfactor] if not isinstance(accfactor, (list, tuple)) else accfactor
        else:
            accfactor = [fs.name for fs in Path(root).iterdir() if fs.is_dir()]
        for acc in accfactor:
            # files
            files = [[
                f for f in fs.iterdir() if f.is_file() and f.suffix == '.mat' and 'mask' not in f.stem
            ] for fs in Path(Path(root) / f'{acc}').iterdir() if fs.is_dir()]

            for fnames in sorted(files):
                for fname in fnames:
                    metadata, num_tframes, num_slices = self._retrieve_metadata(fname)

                    if acquisitions is not None and metadata["acquisition"] not in acquisitions:
                        continue

                    self.examples += [
                        (fname, tframe_ind, slice_ind, metadata) for tframe_ind, slice_ind in np.array(
                            np.meshgrid(range(num_tframes), range(num_slices))
                        ).T.reshape(-1, 2)
                    ]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            # kspace: complex images with the dimensions (sx,sy,sc,sz,t/w) in .mat
            # Load Matlab v7.3 format .mat file using h5py.
            keys = list(hf.keys())
            if len(keys) > 1:
                keys = [k for k in hf.keys() if 'kspace' in k]
            assert len(keys) == 1
            num_tframes, num_slices = hf[keys[0]].shape[0:2]

        metadata = {"acquisition": fname.stem, "num_tframes": num_tframes, "num_slices": num_slices}

        return metadata, num_tframes, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, datatframe, dataslice, metadata = self.examples[i]

        ### load kspace
        if self.tfcontext is not None:
            tframe_no = np.arange(datatframe - self.tfcontext, datatframe + self.tfcontext + 1)
            tframe_no_start = np.maximum(tframe_no, 0).min()
            tframe_no_stop = np.minimum(tframe_no, (metadata["num_tframes"] - 1)).max() + 1
            tframe_prefix = sum(tframe_no < 0)
            tframe_suffix = sum(tframe_no > (metadata["num_tframes"] - 1))
        else:
            tframe_no_start = datatframe
            tframe_no_stop = datatframe + 1
            tframe_prefix = 0
            tframe_suffix = 0
        if self.scontext is not None:
            slice_no = np.arange(dataslice - self.scontext, dataslice + self.scontext + 1)
            slice_no_start = np.maximum(slice_no, 0).min()
            slice_no_stop = np.minimum(slice_no, (metadata["num_slices"] - 1)).max() + 1
            slice_prefix = sum(slice_no < 0)
            slice_suffix = sum(slice_no > (metadata["num_slices"] - 1))
        else:
            slice_no_start = dataslice
            slice_no_stop = dataslice + 1
            slice_prefix = 0
            slice_suffix = 0
        with h5py.File(fname, "r") as hf:
            # kspace: complex images with the dimensions (sx,sy,sc,sz,t/w) in .mat
            # Load Matlab v7.3 format .mat file using h5py (t/w,sz,sc,sy,sx).
            keys = list(hf.keys())
            if len(keys) > 1:
                keys = [k for k in hf.keys() if 'kspace' in k]
            assert len(keys) == 1
            kspace = hf[keys[0]][tframe_no_start:tframe_no_stop, slice_no_start:slice_no_stop]

            # update attrs
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        # handle boundary of tframes and slices
        if tframe_prefix > 0:
            kspace = np.concatenate([np.concatenate([kspace[0:1]] * tframe_prefix, axis=0), kspace], axis=0)
        if tframe_suffix > 0:
            kspace = np.concatenate([kspace, np.concatenate([kspace[-1:]] * tframe_suffix, axis=0)], axis=0)
        if slice_prefix > 0:
            kspace = np.concatenate([np.concatenate([kspace[:,0:1]] * slice_prefix, axis=1), kspace], axis=1)
        if slice_suffix > 0:
            kspace = np.concatenate([kspace, np.concatenate([kspace[:,-1:]] * slice_suffix, axis=1)], axis=1)

        # convert [('real', '<f4'), ('imag', '<f4')] to numpy complex
        kspace = kspace['real'] + 1j*kspace['imag']

        ### load mask if any
        fmasks = [str(fname).replace('.mat', '_mask.mat')]
        fmasks = [fmask for fmask in fmasks if Path(fmask).is_file()]
        if len(fmasks) > 0:
            mask = {}
            for fmask in fmasks:
                with h5py.File(fmask, "r") as hf:
                    # mask: subsampling mask with the dimensions (sx,sy) in .mat
                    # Load Matlab v7.3 format .mat file using h5py (sy,sx).
                    keys = list(hf.keys())
                    if len(keys) > 1:
                        keys = [k for k in hf.keys() if 'mask' in k]
                    assert len(keys) == 1
                    mask[keys[0]] = hf[keys[0]][()].astype(np.float32)
        else:
            mask = None

        ### load target if any
        target = None

        ### update fname for keeping the file structures when saving reconstructions
        # recon_fname = str(fname).split('/ChallengeData/')[1]
        recon_fname = self.challenge + str(fname).split(self.challenge)[1]

        ###
        if self.transform is None:
            sample = (kspace, mask, target, attrs, recon_fname, datatframe, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, recon_fname, datatframe, dataslice)

        return sample
