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

import os
import pathlib
from argparse import ArgumentParser
import distutils.util

import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type
from ktclair.pl_modules import CMRDataModule, KTCLAIRModule
from ktclair.data.transforms import CMRDataTransform


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    if args.kt_model == "KSpaceModule3D" or args.kf_model == "KSpaceModule3D":
        assert len(args.spatial_shrink_size) == 1
    else:
        args.spatial_shrink_size = None
        args.scontext_shrink_size = None
        args.tfcontext_shrink_size = None

    train_transform = CMRDataTransform(
        mask_func=mask,
        use_seed=False,
        spatial_shrink_size=args.spatial_shrink_size,
        scontext_shrink_size=args.scontext_shrink_size,
        tfcontext_shrink_size=args.tfcontext_shrink_size,
        mask_correction_mode=args.mask_correction_mode if str(args.mask_correction_mode) != 'none' else None,
        use_synthetic_target=True,
        enable_crop_size=args.enable_crop_size,
    )
    val_transform = CMRDataTransform(
        mask_func=mask,
        spatial_shrink_size=args.spatial_shrink_size,
        scontext_shrink_size=args.scontext_shrink_size,
        tfcontext_shrink_size=args.tfcontext_shrink_size,
        mask_correction_mode=args.mask_correction_mode if str(args.mask_correction_mode) != 'none' else None,
        use_synthetic_target=True,
        enable_crop_size=args.enable_crop_size,
    )
    test_transform = CMRDataTransform(
        mask_func=mask,
        spatial_shrink_size=args.spatial_shrink_size,
        scontext_shrink_size=args.scontext_shrink_size,
        tfcontext_shrink_size=args.tfcontext_shrink_size,
        mask_correction_mode=args.mask_correction_mode if str(args.mask_correction_mode) != 'none' else None,
        use_synthetic_target=True,
        enable_crop_size=args.enable_crop_size,
    )
    # ptl data module - this handles data loaders
    data_module = CMRDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        task=args.task,
        train_setname=args.train_setname,
        val_setname=args.val_setname,
        test_setname=args.test_setname,
        train_accfactor=args.train_accfactor,
        val_accfactor=args.val_accfactor,
        test_accfactor=args.test_accfactor,
        trainlst=args.trainlst if str(args.trainlst) != 'none' else None,
        vallst=args.vallst if str(args.vallst) != 'none' else None,
        testlst=args.testlst if str(args.testlst) != 'none' else None,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val,
        stats_file=args.stats_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.strategy in ("ddp", "ddp_cpu")),
        acquisitions=args.acquisitions,
        scontext=args.scontext,
        tfcontext=args.tfcontext,
    )

    # ------------
    # model
    # ------------
    model = KTCLAIRModule(
        ### sens_model
        sens_model=args.sens_model,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        mask_center=args.mask_center,
        ### xt_model
        xt_model=args.xt_model,
        xt_num_cascades=args.xt_num_cascades,
        xt_inp_channels=args.xt_inp_channels,
        xt_out_channels=args.xt_out_channels,
        xt_chans=args.xt_chans,
        xt_pools=args.xt_pools,
        xt_dc_mode=args.xt_dc_mode,
        xt_no_parameter_sharing=args.xt_no_parameter_sharing,
        xt_bg_drop_prob=args.xt_bg_drop_prob,
        ### kt_model
        kt_model=args.kt_model,
        kt_num_cascades=args.kt_num_cascades,
        kt_num_blocks=args.kt_num_blocks,
        kt_inp_channels=args.kt_inp_channels,
        kt_out_channels=args.kt_out_channels,
        kt_conv_mode=args.kt_conv_mode,
        kt_kernel_sizes=args.kt_kernel_sizes,
        kt_paddings=args.kt_paddings,
        kt_acti_func=args.kt_acti_func,
        kt_no_parameter_sharing=args.kt_no_parameter_sharing,
        kt_normalize=args.kt_normalize,
        ### xf_model
        xf_model=args.xf_model,
        xf_num_cascades=args.xf_num_cascades,
        xf_inp_channels=args.xf_inp_channels,
        xf_out_channels=args.xf_out_channels,
        xf_chans=args.xf_chans,
        xf_pools=args.xf_pools,
        xf_dc_mode=args.xf_dc_mode,
        xf_no_parameter_sharing=args.xf_no_parameter_sharing,
        xf_bg_drop_prob=args.xf_bg_drop_prob,
        ### kf_model
        kf_model=args.kf_model,
        kf_num_cascades=args.kf_num_cascades,
        kf_num_blocks=args.kf_num_blocks,
        kf_inp_channels=args.kf_inp_channels,
        kf_out_channels=args.kf_out_channels,
        kf_conv_mode=args.kf_conv_mode,
        kf_kernel_sizes=args.kf_kernel_sizes,
        kf_paddings=args.kf_paddings,
        kf_acti_func=args.kf_acti_func,
        kf_no_parameter_sharing=args.kf_no_parameter_sharing,
        kf_normalize=args.kf_normalize,
        ### loss setttings
        loss_num_cascades=args.loss_num_cascades,
        loss_num_slices=args.loss_num_slices,
        loss_names=args.loss_names,
        loss_weights=args.loss_weights,
        loss_decay=args.loss_decay,
        loss_multiscale=args.loss_multiscale,
        use_scan_stats=args.use_scan_stats,
        crop_target=args.crop_target,
        ### optimizer
        optimizer=args.optimizer,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        ### others
        save_keys=args.save_keys,
    )

    # init from pretrained model
    if args.pretrained_model is not None:
        import torch
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
        print('pretrained model loaded!')

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, ckpt_path=args.resume_from_checkpoint, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # # basic args
    # path_config = pathlib.Path("../../fastmri_dirs.yaml")
    # backend = "ddp"
    # num_gpus = 2 if backend == "ddp" else 1
    # batch_size = 1

    # # set defaults based on optional directory config
    # data_path = fetch_dir("knee_path", path_config)
    # default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--spatial_shrink_size", nargs="+", default=[0], type=int,
        help="spatial_shrink_size.",
    )
    parser.add_argument(
        "--scontext_shrink_size", nargs="+", default=[0], type=int,
        help="scontext_shrink_size.",
    )
    parser.add_argument(
        "--tfcontext_shrink_size", nargs="+", default=[0], type=int,
        help="tfcontext_shrink_size.",
    )

    # pretrained model
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="Path to the checkpoint of the pretrained model for warm-up",
    )

    #
    parser.add_argument(
        "--mask_correction_mode", nargs="+", default=None, type=str,
        help="",
    )

    parser.add_argument(
        "--enable_crop_size",
        default=False,
        type=lambda x:bool(distutils.util.strtobool(x)),
        help="",
    )

    # data config
    parser = CMRDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        # data_path=data_path,  # path to fastMRI data
        # mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        # challenge="multicoil",  # only multicoil implemented for VarNet
        # batch_size=batch_size,  # number of samples per batch
        # test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = KTCLAIRModule.add_model_specific_args(parser)
    parser.set_defaults(
        ### sens_model
        sens_model="Sensitivity3DModule",
        sens_chans=8,
        sens_pools=4,
        mask_center=True,
        ### xt_model
        xt_model="NormUnet3D",
        xt_num_cascades=12,
        xt_inp_channels=2,
        xt_out_channels=2,
        xt_chans=48,
        xt_pools=4,
        xt_dc_mode='GD',
        xt_no_parameter_sharing=True,
        xt_bg_drop_prob=None,
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
        xf_chans=48,
        xf_pools=4,
        xf_dc_mode='GD',
        xf_no_parameter_sharing=True,
        xf_bg_drop_prob=None,
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
        ### loss setttings
        loss_num_cascades=1,
        loss_num_slices=1,
        loss_names=["ssim", "l1", "xt", "roi", "kt", "acs"], # ["ssim", "l1", "xt", "roi", "kt", "acs"]
        loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_decay=False,
        loss_multiscale=[0.5, 0.75, 1.0, 1.25, 1.5],
        use_scan_stats=True,
        ### optimizer
        optimizer='Adam', # optimization algorithm
        lr=0.0003, # Adam learning rate
        lr_step_size=40, # epoch at which to decrease learning rate
        lr_gamma=0.1, # extent to which to decrease learning rate
        weight_decay=0.0, # weight regularization strength
        momentum=0.99, # SGD momentum factor
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        # gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        # accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
        # default_root_dir=default_root_dir,  # directory for logs and checkpoints
        # max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    args.default_root_dir = pathlib.Path(args.default_root_dir)
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    # set resume_from_checkpoint to None if the pretrained model was provided
    if args.pretrained_model is not None:
        args.resume_from_checkpoint = None

    return args


def run_cli():
    args = build_args()
    import pprint
    pp = pprint.PrettyPrinter(indent=4, compact=True)
    pp.pprint(args)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
