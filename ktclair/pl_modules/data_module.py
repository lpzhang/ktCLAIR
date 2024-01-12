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
from pathlib import Path
from typing import Callable, Optional, Union, Sequence, Tuple

import pytorch_lightning as pl
import torch
from ktclair.data.volume_sampler import VolumeSampler
from ktclair.data.cmr_data import CombinedCMRDataset, CMRDataset

import distutils.util


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        CMRDataset, CombinedCMRDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedCMRDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2 ** 32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2 ** 32 - 1))


class CMRDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        task: str,
        train_setname: str,
        val_setname: str,
        test_setname: str,
        train_accfactor: Sequence[str],
        val_accfactor: Sequence[str],
        test_accfactor: Sequence[str],
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        stats_file: Optional[Path] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        acquisitions: Optional[Sequence[str]] = None,
        scontext: Optional[int] = None,
        tfcontext: Optional[int] = None,
        trainlst: Optional[Path] = None,
        vallst: Optional[Path] = None,
        testlst: Optional[Path] = None,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
            acquisitions: Optional; If provided, only slices with the desired
                MRI pulse sequence will be considered.
        """
        super().__init__()

        self.data_path = data_path
        self.challenge = challenge
        self.task = task
        self.train_setname = train_setname
        self.val_setname = val_setname
        self.test_setname = test_setname
        self.train_accfactor = train_accfactor
        self.val_accfactor = val_accfactor
        self.test_accfactor = test_accfactor
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.stats_file = stats_file
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.acquisitions = acquisitions
        self.scontext = scontext
        self.tfcontext = tfcontext
        self.trainlst = trainlst
        self.vallst = vallst
        self.testlst = testlst

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            scontext = self.scontext
            tfcontext = self.tfcontext
        else:
            is_train = False
            scontext = self.scontext
            tfcontext = self.tfcontext

        # if desired, combine train and val together for the train split
        dataset: Union[CMRDataset, CombinedCMRDataset]
        if is_train and self.combine_train_val:
            # data_paths = [
            #     self.data_path / f"{self.challenge}_train",
            #     self.data_path / f"{self.challenge}_val",
            # ]
            data_paths = [
                self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.train_setname}",
                self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.val_setname}",
            ]
            accfactors = [
                self.train_accfactor,
                self.val_accfactor,
            ]
            filelsts = [
                self.trainlst,
                self.vallst,
            ]
            data_partitions = ['train', 'val']
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            tasks = [self.task, self.task]
            stats_files = [self.stats_file, self.stats_file]
            acquisitions = [self.acquisitions, self.acquisitions] if self.acquisitions is not None else None
            scontexts = [self.scontext, self.scontext] if self.scontext is not None else None
            tfcontexts = [self.tfcontext, self.tfcontext] if self.tfcontext is not None else None

            dataset = CombinedCMRDataset(
                roots=data_paths,
                accfactors=accfactors,
                filelsts=filelsts,
                data_partitions=data_partitions,
                transforms=data_transforms,
                challenges=challenges,
                tasks=tasks,
                stats_files=stats_files,
                use_dataset_cache=self.use_dataset_cache_file,
                acquisitions=acquisitions,
                scontexts=scontexts,
                tfcontexts=tfcontexts,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                # data_path = self.data_path / f"{self.challenge}_{data_partition}"
                if data_partition == "train":
                    data_path = self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.train_setname}"
                    accfactor = self.train_accfactor
                    filelst = self.trainlst
                elif data_partition == "val":
                    data_path = self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.val_setname}"
                    accfactor = self.val_accfactor
                    filelst = self.vallst
                else:
                    data_path = self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.test_setname}"
                    accfactor = self.test_accfactor
                    filelst = self.testlst

            dataset = CMRDataset(
                root=data_path,
                accfactor=accfactor,
                filelst=filelst,
                data_partition=data_partition,
                transform=data_transform,
                stats_file=self.stats_file,
                challenge=self.challenge,
                task=self.task,
                use_dataset_cache=self.use_dataset_cache_file,
                acquisitions=self.acquisitions,
                scontext=scontext,
                tfcontext=tfcontext,
            )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                # test_path = self.data_path / f"{self.challenge}_test"
                test_path = self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.test_setname}"
            data_paths = [
                self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.train_setname}",
                self.data_path / f"{self.challenge}" / f"{self.task}" / f"{self.val_setname}",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            accfactors = [
                self.train_accfactor,
                self.val_accfactor,
                self.test_accfactor,
            ]
            filelsts = [
                self.trainlst,
                self.vallst,
                self.testlst,
            ]
            data_partitions = ['train', 'val', 'test']
            for i, (data_path, data_transform, accfactor, filelst, data_partition) in enumerate(
                zip(data_paths, data_transforms, accfactors, filelsts, data_partitions)
            ):
                scontext = self.scontext
                tfcontext = self.tfcontext
                _ = CMRDataset(
                    root=data_path,
                    accfactor=accfactor,
                    filelst=filelst,
                    data_partition=data_partition,
                    transform=data_transform,
                    stats_file=self.stats_file,
                    challenge=self.challenge,
                    task=self.task,
                    use_dataset_cache=self.use_dataset_cache_file,
                    acquisitions=self.acquisitions,
                    scontext=scontext,
                    tfcontext=tfcontext,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(
            self.val_transform, data_partition="val",
        )

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform,
            data_partition=self.test_split,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--challenge",
            choices=("SingleCoil", "MultiCoil"),
            default="SingleCoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--task",
            choices=("Cine", "Mapping"),
            default="Mapping",
            type=str,
            help="Which task to preprocess for",
        )
        parser.add_argument(
            "--train_setname",
            default=None,
            type=str,
            help="TrainingSet",
        )
        parser.add_argument(
            "--val_setname",
            default=None,
            type=str,
            help="ValidationSet",
        )
        parser.add_argument(
            "--test_setname",
            default=None,
            type=str,
            help="TestSet",
        )
        parser.add_argument(
            "--train_accfactor",
            nargs="+",
            default=["AccFactor04", "AccFactor08", "AccFactor10",],
            type=str,
            help="AccFactor04, AccFactor08, AccFactor10",
        )
        parser.add_argument(
            "--val_accfactor",
            nargs="+",
            default=["AccFactor04", "AccFactor08", "AccFactor10",],
            type=str,
            help="AccFactor04, AccFactor08, AccFactor10",
        )
        parser.add_argument(
            "--test_accfactor",
            nargs="+",
            default=["AccFactor04", "AccFactor08", "AccFactor10",],
            type=str,
            help="AccFactor04, AccFactor08, AccFactor10",
        )
        parser.add_argument(
            "--trainlst",
            default=None,
            type=Path,
            help="trainlst",
        )
        parser.add_argument(
            "--vallst",
            default=None,
            type=Path,
            help="vallst",
        )
        parser.add_argument(
            "--testlst",
            default=None,
            type=Path,
            help="testlst",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--stats_file",
            default=None,
            type=Path,
            help="Path to data statistics",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=lambda x:bool(distutils.util.strtobool(x)),
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        # custom arguments
        parser.add_argument(
            "--acquisitions",
            nargs="+",
            default=["cine_lax", "cine_sax", "T1map", "T2map"],
            type=str,
            help="Which pulse sequence data to use. If not given all will be used.",
        )
        parser.add_argument(
            "--scontext", default=0, type=int, help="context in slices"
        )
        parser.add_argument(
            "--tfcontext", default=0, type=int, help="context in tframes"
        )

        return parser
