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
import nibabel as nib


class CombinedCMRDataset(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        challenges: Sequence[str],
        tasks: Sequence[str],
        accfactors: Sequence[Sequence[str]],
        data_partitions: Sequence[str],
        filelsts: Optional[Sequence[Union[str, Path, os.PathLike]]] = None,
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        stats_files: Optional[Sequence[Union[str, Path, os.PathLike]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        acquisitions: Optional[Sequence[Optional[Sequence[str]]]] = None,
        scontexts: Optional[Tuple[int]] = None,
        tfcontexts: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            roots: Paths to the datasets.
            challenges: "singlecoil" or "multicoil" depending on which
                challenge to use.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form. The transform
                function should take 'kspace', 'target', 'attributes',
                'filename', and 'slice' as inputs. 'target' may be null for
                test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            acquisitions: Optional; If provided, only slices with the desired
                MRI pulse sequences will be considered.
        """
        if transforms is None:
            transforms = [None] * len(roots)
        if acquisitions is None:
            acquisitions = [None] * len(roots)
        if scontexts is None:
            scontexts = [None] * len(roots)
        if tfcontexts is None:
            tfcontexts = [None] * len(roots)
        if not (
            len(roots)
            == len(transforms)
            == len(challenges)
            == len(tasks)
            == len(accfactors)
            == len(data_partitions)
            == len(filelsts)
            == len(stats_files)
            == len(acquisitions)
            == len(scontexts)
            == len(tfcontexts)
        ):
            raise ValueError(
                "Lengths of roots, transforms, challenges do not match"
            )

        self.datasets = []
        self.examples: List[Tuple[Path, int, Dict[str, object]]] = []
        for i in range(len(roots)):
            self.datasets.append(
                CMRDataset(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    task=tasks[i],
                    accfactor=accfactors[i],
                    data_partition=data_partitions[i],
                    filelst=filelsts[i],
                    stats_file=stats_files[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    acquisitions=acquisitions[i],
                    scontext=scontexts[i],
                    tfcontext=tfcontexts[i],
                )
            )

            self.examples = self.examples + self.datasets[-1].examples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class CMRDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        task: str,
        accfactor: Sequence[str],
        data_partition: str,
        filelst: Optional[Union[str, Path, os.PathLike]] = None,
        transform: Optional[Callable] = None,
        stats_file: Optional[Union[str, Path, os.PathLike]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
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
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            acquisitions: Optional; If provided, only slices with the desired
                MRI pulse sequence will be considered.
        """
        if challenge not in ("SingleCoil", "MultiCoil"):
            raise ValueError('challenge should be either "SingleCoil" or "MultiCoil"')
        self.challenge = challenge

        if task not in ("Cine", "Mapping"):
            raise ValueError('task should be either "Cine" or "Mapping"')
        self.task = task

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        # self.recons_key = (
        #     "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        # )
        self.recons_key = 'img4ranking'

        self.examples = []

        self.scontext = scontext
        self.tfcontext = tfcontext

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        dataset_cache_key = Path(root) / f'{data_partition}'
        # if dataset_cache.get(root) is None or not use_dataset_cache:
        if dataset_cache.get(dataset_cache_key) is None or not use_dataset_cache:
            # get file list
            if filelst is not None:
                with open(filelst, 'r') as flst:
                    flst = [f.rstrip() for f in flst] # All lines including the blank ones
                    flst = [f for f in flst if f] # Non-blank lines
                if len(flst) == 0:
                    flst = None
            else:
                flst = None
            for acc in accfactor:
                # files
                if flst is None:
                    files = [[
                        f for f in fs.iterdir() if f.is_file() and f.suffix == '.mat' and 'mask' not in f.stem
                    ] for fs in Path(Path(root) / f'{acc}').iterdir() if fs.is_dir()]
                else:
                    files = [[
                        f for f in fs.iterdir() if f.is_file() and f.suffix == '.mat' and 'mask' not in f.stem
                    ] for fs in Path(Path(root) / f'{acc}').iterdir() if fs.is_dir() and fs.name in flst]

                for fnames in sorted(files):
                    for fname in fnames:
                        metadata, num_tframes, num_slices = self._retrieve_metadata(fname)

                        if acquisitions is not None and metadata["acquisition"] not in acquisitions:
                            continue

                        self.examples += [
                            (fname, tframe_ind, slice_ind, metadata, 'supervised', acc) for tframe_ind, slice_ind in np.array(
                                np.meshgrid(range(num_tframes), range(num_slices))
                            ).T.reshape(-1, 2)
                        ]

            # if dataset_cache.get(root) is None and use_dataset_cache:
            if dataset_cache.get(dataset_cache_key) is None and use_dataset_cache:
                # dataset_cache[root] = self.examples
                dataset_cache[dataset_cache_key] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            # self.examples = dataset_cache[root]
            self.examples = dataset_cache[dataset_cache_key]

        # load statistics if we have
        if stats_file is not None and Path(stats_file).exists():
            with open(stats_file, "rb") as f:
                self.stats = pickle.load(f)
        else:
            self.stats = None

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            # kspace: complex images with the dimensions (sx,sy,sc,sz,t/w) in .mat
            # Load Matlab v7.3 format .mat file using h5py.
            keys = [k for k in hf.keys() if 'kspace' in k]
            assert len(keys) == 1
            # num_tframes, num_slices = hf["kspace_full"].shape[0:2]
            num_tframes, num_slices = hf[keys[0]].shape[0:2]

        ### load the inversion time for MOLLI (ms) or the echo time for T2prep-SSFP (ms)
        csvfile = fname.with_suffix('.csv')
        if csvfile.exists():
            seq_params = pd.read_csv(csvfile, index_col=0).dropna(axis=1)
            seq_params = seq_params.sort_index(
                key=lambda x: x.str.lower(), axis=0
            ).sort_index(key=lambda x: x.str.lower(), axis=1).to_numpy()
            # T2 only has one column for all slices, repeat the column to match the `num_slices'
            if seq_params.shape[1] == 1:
                seq_params = seq_params.repeat(num_slices, axis=1)
            assert(num_tframes == seq_params.shape[0])
            assert(num_slices == seq_params.shape[1])
        else:
            seq_params = None

        metadata = {"acquisition": fname.stem, "num_tframes": num_tframes, "num_slices": num_slices, 'seq_params': seq_params}

        return metadata, num_tframes, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, datatframe, dataslice, metadata, datalabel, accfactor = self.examples[i]

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
        with h5py.File(
            fname if 'TrainingSet' not in str(fname) else str(fname).replace(accfactor, 'FullSample'),
            "r",
        ) as hf:
            # kspace: complex images with the dimensions (sx,sy,sc,sz,t/w) in .mat
            # Load Matlab v7.3 format .mat file using h5py (t/w,sz,sc,sy,sx).
            keys = [k for k in hf.keys() if 'kspace' in k]
            assert len(keys) == 1
            kspace = hf[keys[0]][tframe_no_start:tframe_no_stop, slice_no_start:slice_no_stop]

            # update attrs
            attrs = dict(hf.attrs)
            attrs.update(metadata)

            # recons_key
            # recons_key = keys[0]
            # attrs.update({"recons_key": recons_key})
            attrs.update({"recons_key": self.recons_key})

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

        # update seq_params if any
        if attrs['seq_params'] is not None:
            seq_params = attrs['seq_params'][tframe_no_start:tframe_no_stop, slice_no_start:slice_no_stop]
            # handle boundary of tframes and slices
            if tframe_prefix > 0:
                seq_params = np.concatenate([np.concatenate([seq_params[0:1]] * tframe_prefix, axis=0), seq_params], axis=0)
            if tframe_suffix > 0:
                seq_params = np.concatenate([seq_params, np.concatenate([seq_params[-1:]] * tframe_suffix, axis=0)], axis=0)
            if slice_prefix > 0:
                seq_params = np.concatenate([np.concatenate([seq_params[:,0:1]] * slice_prefix, axis=1), seq_params], axis=1)
            if slice_suffix > 0:
                seq_params = np.concatenate([seq_params, np.concatenate([seq_params[:,-1:]] * slice_suffix, axis=1)], axis=1)
            attrs['seq_params'] = seq_params

        ### load mask if any
        if "FullSample" in str(fname):
            fmasks = [str(fname).replace('FullSample', acc).replace('.mat', '_mask.mat') for acc in ("AccFactor04", "AccFactor08", "AccFactor10")]
        else:
            fmasks = [str(fname).replace('.mat', '_mask.mat')]
        fmasks = [fmask for fmask in fmasks if Path(fmask).is_file()]
        if len(fmasks) > 0:
            mask = {}
            for fmask in fmasks:
                with h5py.File(fmask, "r") as hf:
                    # mask: subsampling mask with the dimensions (sx,sy) in .mat
                    # Load Matlab v7.3 format .mat file using h5py (sy,sx).
                    keys = [k for k in hf.keys() if 'mask' in k]
                    assert len(keys) == 1
                    mask[keys[0]] = hf[keys[0]][()].astype(np.float32)
        else:
            mask = None

        ### load target if any
        target = None

        ### load seg if any
        if 'TrainingSet' in str(fname) and  'MultiCoil' in str(fname):
            seg = Path(str(fname).replace('MultiCoil', 'SingleCoil').replace(accfactor, 'SegmentROI').replace('.mat', '_label.nii.gz'))
            seg = nib.load(seg).slicer[...,slice_no_start:slice_no_stop] if seg.exists() else None
            seg = seg.get_fdata().astype(np.uint8).transpose([2,1,0])[None] if seg is not None else None # (sx, sy, sz) to (1, sz, sy, sx)
        else:
            seg = None

        ### update fname for keeping the file structures when saving reconstructions
        # recon_fname = str(fname).split('/ChallengeData/')[1]
        recon_fname = self.challenge + str(fname).split(self.challenge)[1]

        ### stats
        scan_id = recon_fname if 'TrainingSet' not in str(recon_fname) else str(recon_fname).replace(accfactor, 'FullSample')
        scan_stats = self.stats.get(scan_id, None) if self.stats is not None else None

        ###
        if self.transform is None:
            sample = (kspace, mask, target, seg, attrs, recon_fname, datatframe, dataslice, datalabel, scan_stats)
        else:
            sample = self.transform(kspace, mask, target, seg, attrs, recon_fname, datatframe, dataslice, datalabel, scan_stats)

        return sample
