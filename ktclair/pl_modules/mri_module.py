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

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import fastmri
import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from torchmetrics.metric import Metric

from typing import Optional, Sequence
import h5py
import hdf5storage
from einops import rearrange


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16, save_keys: Optional[Sequence[str]] = None):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

        # save results
        self.save_keys = save_keys

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "tframe_num",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

        # log extra images to tensorboard
        self.log_extra_images(val_logs)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            tframe_num = int(val_logs["tframe_num"][i].cpu())
            slice_num = int(val_logs["slice_num"][i].cpu())
            example_num = f"t{tframe_num}-s{slice_num}"
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            # normalization using the max_value of image volume
            target = target / maxval
            output = output / maxval
            data_range = np.maximum(target.max(), output.max()) - np.minimum(target.min(), output.min())
            # data_range = maxval

            mse_vals[fname][example_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][example_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            # ssim_vals[fname][example_num] = torch.tensor(
            #     evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            # ).view(1)
            ssim_vals[fname][example_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=data_range)
            ).view(1)
            psnr_vals[fname][example_num] = torch.tensor(
                evaluate.psnr(target[None, ...], output[None, ...], maxval=data_range)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "psnr_vals": dict(psnr_vals),
            "max_vals": max_vals,
        }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def log_extra_images(self, val_logs):
        # keys of extra images
        extra_keys = [
            extra_key for extra_key in val_logs.keys() if extra_key not in (
                "batch_idx",
                "fname",
                "tframe_num",
                "slice_num",
                "max_value",
                "output",
                "target",
                "val_loss",
            )
        ]

        # check extra images
        for extra_key in extra_keys:
            if val_logs[extra_key].ndim == 2:
                val_logs[extra_key] = val_logs[extra_key].unsqueeze(0)
            elif val_logs[extra_key].ndim != 3:
                raise RuntimeError(f"Unexpected {extra_key} size from validation_step.")

        # log extra images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                for extra_key in extra_keys:
                    extra_image = val_logs[extra_key][i].unsqueeze(0)
                    extra_image = extra_image / extra_image.max()
                    self.log_image(f"{key}/{extra_key}", extra_image)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["psnr_vals"].keys():
                psnr_vals[k].update(val_log["psnr_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == psnr_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            # metrics["psnr"] = (
            #     metrics["psnr"]
            #     + 20
            #     * torch.log10(
            #         torch.tensor(
            #             max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
            #         )
            #     )
            #     - 10 * torch.log10(mse_val)
            # )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["psnr"] = metrics["psnr"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    # def test_epoch_end(self, test_logs):
    #     outputs = defaultdict(dict)

    #     # use dicts for aggregation to handle duplicate slices in ddp mode
    #     for log in test_logs:
    #         for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
    #             outputs[fname][int(slice_num.cpu())] = log["output"][i]

    #     # stack all the slices for each file
    #     for fname in outputs:
    #         outputs[fname] = np.stack(
    #             [out for _, out in sorted(outputs[fname].items())]
    #         )

    #     # pull the default_root_dir if we have a trainer, otherwise save to cwd
    #     if hasattr(self, "trainer"):
    #         save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
    #     else:
    #         save_path = pathlib.Path.cwd() / "reconstructions"
    #     self.print(f"Saving reconstructions to {save_path}")

    #     fastmri.save_reconstructions(outputs, save_path)

    def test_step_end(self, test_logs):
        # check inputs
        for k in (
            "fname",
            "tframe_num",
            "slice_num",
            "max_value",
            "recons_key",
            "output",
            "target",
        ):
            if k not in test_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by test_step."
                )
        if test_logs["output"].ndim == 2:
            test_logs["output"] = test_logs["output"].unsqueeze(0)
        elif test_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from test_step.")
        if test_logs["target"].ndim == 2:
            test_logs["target"] = test_logs["target"].unsqueeze(0)
        elif test_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from test_step.")

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(test_logs["fname"]):
            tframe_num = int(test_logs["tframe_num"][i].cpu())
            slice_num = int(test_logs["slice_num"][i].cpu())
            example_num = f"t{tframe_num}-s{slice_num}"
            maxval = test_logs["max_value"][i].cpu().numpy()
            output = test_logs["output"][i].cpu().numpy()
            target = test_logs["target"][i].cpu().numpy()

            # normalization using the max_value of image volume
            target = target / maxval
            output = output / maxval
            data_range = np.maximum(target.max(), output.max()) - np.minimum(target.min(), output.min())
            # data_range = maxval

            mse_vals[fname][example_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][example_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            # ssim_vals[fname][example_num] = torch.tensor(
            #     evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            # ).view(1)
            ssim_vals[fname][example_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=data_range)
            ).view(1)
            psnr_vals[fname][example_num] = torch.tensor(
                evaluate.psnr(target[None, ...], output[None, ...], maxval=data_range)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "fname": test_logs["fname"],
            "tframe_num": test_logs["tframe_num"],
            "slice_num": test_logs["slice_num"],
            "recons_key": test_logs["recons_key"],
            "output": test_logs["output"].cpu().numpy() if "pred" in self.save_keys else None,
            "target": test_logs["target"].cpu().numpy() if "target" in self.save_keys else None,
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "psnr_vals": dict(psnr_vals),
            "max_vals": max_vals,
        }


    def test_epoch_end(self, test_logs):
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for test_log in test_logs:
            for k in test_log["mse_vals"].keys():
                mse_vals[k].update(test_log["mse_vals"][k])
            for k in test_log["target_norms"].keys():
                target_norms[k].update(test_log["target_norms"][k])
            for k in test_log["ssim_vals"].keys():
                ssim_vals[k].update(test_log["ssim_vals"][k])
            for k in test_log["psnr_vals"].keys():
                psnr_vals[k].update(test_log["psnr_vals"][k])
            for k in test_log["max_vals"]:
                max_vals[k] = test_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == psnr_vals.keys()
            == max_vals.keys()
        )

        # # sort and log all the slices for each file
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()
        # for fname in sorted(ssim_vals.keys()):
        #     # log each slice
        #     for slice_no, _ in sorted(ssim_vals[fname].items()):
        #         writer.add_scalar(f"test_metrics/{fname}/ssim/slice", ssim_vals[fname][slice_no], slice_no)
        #         writer.add_scalar(f"test_metrics/{fname}/psnr/slice", psnr_vals[fname][slice_no], slice_no)

        #     # log each file
        #     writer.add_scalar(f"test_metrics/{fname}/ssim", torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])))
        #     writer.add_scalar(f"test_metrics/{fname}/psnr", torch.mean(torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])))
        # writer.close()

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0

        # scan_metrics
        scan_metrics = dict()
        for metric_name in ("nmse", "ssim", "psnr"):
            scan_metrics[metric_name] = dict()

        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            # metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            # # metrics["psnr"] = (
            # #     metrics["psnr"]
            # #     + 20
            # #     * torch.log10(
            # #         torch.tensor(
            # #             max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
            # #         )
            # #     )
            # #     - 10 * torch.log10(mse_val)
            # # )
            # metrics["ssim"] = metrics["ssim"] + torch.mean(
            #     torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            # )
            # metrics["psnr"] = metrics["psnr"] + torch.mean(
            #     torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            # )

            scan_metrics["nmse"][fname] = mse_val / target_norm
            scan_metrics["ssim"][fname] = torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            scan_metrics["psnr"][fname] = torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

            metrics["nmse"] = metrics["nmse"] + scan_metrics["nmse"][fname]
            metrics["ssim"] = metrics["ssim"] + scan_metrics["ssim"][fname]
            metrics["psnr"] = metrics["psnr"] + scan_metrics["psnr"][fname]

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))

        for metric, value in metrics.items():
            self.log(f"test_metrics/{metric}", value / tot_examples, prog_bar=True)

        # save results
        if self.save_keys is not None:
            # Write details to file
            if "scan_metric" in self.save_keys:
                with open('scan_metrics.csv', "w") as f:
                    for metric_name in scan_metrics.keys():
                        for name, value in scan_metrics[metric_name].items():
                            f.write("{},{},{}\n".format(metric_name, name, value))

            # Write Reconstruction
            outputs = defaultdict(dict) if "pred" in self.save_keys else None
            targets = defaultdict(dict) if "target" in self.save_keys else None
            recons_keys = defaultdict(set) if "pred" in self.save_keys else None

            # use dicts for aggregation to handle duplicate slices in ddp mode
            if outputs is not None:
                for log in test_logs:
                    for i, (fname, tframe_num, slice_num, recons_key) in enumerate(
                        zip(log["fname"], log["tframe_num"], log["slice_num"], log["recons_key"],)
                    ):
                        example_num = f"t{int(tframe_num.cpu())}-s{int(slice_num.cpu())}"
                        # output
                        output = log["output"][i]
                        if output.shape[-1] == 2:
                            output = output[...,0] + 1j*output[...,1]
                        outputs[fname][example_num] = output
                        # target
                        if targets is not None:
                            target = log["target"][i]
                            if target.shape[-1] == 2:
                                target = target[...,0] + 1j*target[...,1]
                            targets[fname][example_num] = target
                        # recons_key
                        recons_keys[fname].add(recons_key)

                # stack all the slices for each file
                # for fname in outputs:
                #     outputs[fname] = np.stack(
                #         [out for _, out in sorted(outputs[fname].items())]
                #     )
                #     if targets is not None:
                #         targets[fname] = np.stack(
                #             [out for _, out in sorted(targets[fname].items())]
                #         )
                for fname in outputs:
                    # example_keys = sorted(outputs[fname].keys(), key=lambda k: (int(k[0].split('-')[0].split('t')[1]), int(k[0].split('-')[1].split('s')[1])))
                    # num_tframes and num_slices
                    num_tframes = len(set([k.split('-')[0] for k in outputs[fname].keys()]))
                    num_slices  = len(set([k.split('-')[1] for k in outputs[fname].keys()]))
                    # outputs
                    outputs[fname] = np.stack(
                        [out for _, out in sorted(
                            outputs[fname].items(),
                            key=lambda k: (int(k[0].split('-')[0].split('t')[1]), int(k[0].split('-')[1].split('s')[1]))
                        )]
                    )
                    outputs[fname] = outputs[fname].reshape((num_tframes, num_slices) + outputs[fname].shape[1:])
                    # targets if any
                    if targets is not None:
                        targets[fname] = np.stack(
                            [out for _, out in sorted(
                                targets[fname].items(),
                                key=lambda k: (int(k[0].split('-')[0].split('t')[1]), int(k[0].split('-')[1].split('s')[1]))
                            )]
                        )
                        targets[fname] = targets[fname].reshape((num_tframes, num_slices) + targets[fname].shape[1:])
                    # recons_key
                    assert len(recons_keys[fname]) == 1
                    recons_keys[fname] = recons_keys[fname].pop()

                # pull the default_root_dir if we have a trainer, otherwise save to cwd
                if hasattr(self, "trainer"):
                    save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
                else:
                    save_path = pathlib.Path.cwd() / "reconstructions"
                self.print(f"Saving reconstructions to {save_path}")

                # save as the HDF5 data format
                # save_path.mkdir(exist_ok=True, parents=True)
                for fname, recons in outputs.items():
                    out_fpath = save_path / fname
                    out_fpath.parent.mkdir(exist_ok=True, parents=True)
                    # with h5py.File(out_fpath, "w") as hf:
                    #     hf.create_dataset(f"{recons_keys[fname]}", data=recons)
                    #     if targets is not None:
                    #         hf.create_dataset("target", data=targets[fname])
                    #
                    # convert to have the same dimension order as the original data when using hdf5storage.savemat
                    recons = rearrange(recons, 'tf sz sy sx -> sx sy sz tf')
                    if targets is None:
                        hdf5storage.savemat(str(out_fpath), {recons_keys[fname]:recons})
                    else:
                        target = rearrange(targets[fname], 'tf sz sy sx -> sx sy sz tf')
                        hdf5storage.savemat(str(out_fpath), {recons_keys[fname]:recons, "target":target})


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )
        # save results
        parser.add_argument(
            "--save_keys",
            nargs="+",
            default=["scan_metric",],
            type=str,
            help="Which result to save",
        )

        return parser
