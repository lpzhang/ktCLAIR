#!/bin/bash

#SBATCH --job-name=example
#SBATCH --output=example.log
#
#SBATCH --ntasks=1
#SBATCH --time=7200
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=deployment

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_P2P_DISABLE=1

# cuda and cudnn paths
export PATH=$HOME/.usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export CUDADIR=$HOME/.usr/local/cuda-11.0

# conda and python envs
conda_env=$HOME/.usr/local/miniconda3/envs/fastMRI
export LD_LIBRARY_PATH=${conda_env}/lib:$LD_LIBRARY_PATH
echo --conda-env=${conda_env}

# PYTHONPATH is an environment variable which you can set to 
# add additional directories where python will look for modules and packages.
export PYTHONPATH=$HOME/dev/ktCLAIR
echo --PYTHONPATH=${PYTHONPATH}

# project infos
root_dir=$HOME/dev/ktCLAIR
data_path=${root_dir}/datasets/CMRxRecon/MICCAIChallenge2023/ChallengeData
func_path=${root_dir}/ktclair_examples/ktclair_cmr.py
default_root_dir=./
echo --root_dir=${root_dir}
echo --data_path=${data_path}
echo --func_path=${func_path}
echo --default_root_dir=${default_root_dir}
echo --script=$0

# data params
acquisitions=(cine_lax cine_sax T1map T2map)
# challenge=SingleCoil/MultiCoil
challenge=MultiCoil
# task=Cine/Mapping
task=Cine
# TrainingSet/ValidationSet
train_setname=TrainingSet
val_setname=TrainingSet
test_setname=ValidationSet
# AccFactor04  AccFactor08  AccFactor10  FullSample
train_accfactor=(FullSample)
val_accfactor=(FullSample)
test_accfactor=(AccFactor04  AccFactor08  AccFactor10)
trainlst=${root_dir}/ktclair_examples/datasplitlists/train.lst.seed42.s80s20
vallst=${root_dir}/ktclair_examples/datasplitlists/val.lst.seed42.s80s20
testlst=none
# statistics
stats_file=${root_dir}/ktclair_examples/dataset_statistics/raw_data_stats.pkl
mask_type=(equispaced)
center_fractions=(0.08 0.08 0.08)
accelerations=(4 8 10)
scontext=0
tfcontext=1
spatial_shrink_size=(4)
scontext_shrink_size=(0)
tfcontext_shrink_size=(1)
combine_train_val=True
mask_correction_mode=(none) # (sy sx)/(none)
echo --acquisitions=${acquisitions[*]}
echo --challenge=${challenge}
echo --task=${task}
echo --train_setname=${train_setname}
echo --val_setname=${val_setname}
echo --test_setname=${test_setname}
echo --train_accfactor=${train_accfactor}
echo --val_accfactor=${val_accfactor}
echo --test_accfactor=${test_accfactor}
echo --trainlst=${trainlst}
echo --vallst=${vallst}
echo --testlst=${testlst}
echo --stats_file=${stats_file}
echo --mask_type=${mask_type}
echo --center_fractions=${center_fractions}
echo --accelerations=${accelerations}
echo --scontext=${scontext}
echo --tfcontext=${tfcontext}
echo --spatial_shrink_size=${spatial_shrink_size[*]}
echo --scontext_shrink_size=${scontext_shrink_size[*]}
echo --tfcontext_shrink_size=${tfcontext_shrink_size[*]}
echo --combine_train_val=${combine_train_val}
echo --mask_correction_mode=${mask_correction_mode}

# network params
### xt_model
xt_model=NormUnet3DXF # NormUnet3D/NormUnet3DXF/none
xt_num_cascades=12
xt_chans=32
xt_pools=4
### kt_model
kt_model=KSpaceModule3D # KSpaceModule3D/none
kt_num_cascades=12
kt_num_blocks=4
kt_normalize=True
### xf_model
xf_model=none # NormUnet3D/none
xf_num_cascades=12
xf_chans=48
xf_pools=4
### kf_model
kf_model=none # KSpaceModule3D/none
kf_num_cascades=12
kf_num_blocks=4
kf_normalize=True
### loss setttings
loss_num_cascades=1
loss_num_slices=1
loss_names=(ssim l1 xt acs) # (ssim l1 xt roi kt acs)
loss_weights=(1.0 1.0 1.0 1.0) # (1.0 1.0 1.0 1.0 1.0 1.0)
loss_decay=False
loss_multiscale=(1.0) # (0.5 0.75 1.0 1.25 1.5)
enable_crop_size=False
crop_target=False
### optimizer
optimizer=Adam # Adam/SGD/AdamW
lr=0.0003 # Adam learning rate
lr_step_size=40 # epoch at which to decrease learning rate
lr_gamma=0.1 # extent to which to decrease learning rate
weight_decay=0.0 # weight regularization strength
momentum=0.99 # SGD momentum factor
### others
save_keys=(scan_metric pred)

echo --xt_model=${xt_model}
echo --xt_num_cascades=${xt_num_cascades}
echo --xt_chans=${xt_chans}
echo --xt_pools=${xt_pools}
echo --kt_model=${kt_model}
echo --kt_num_cascades=${kt_num_cascades}
echo --kt_num_blocks=${kt_num_blocks}
echo --kt_normalize=${kt_normalize}
echo --xf_model=${xf_model}
echo --xf_num_cascades=${xf_num_cascades}
echo --xf_chans=${xf_chans}
echo --xf_pools=${xf_pools}
echo --kf_model=${kf_model}
echo --kf_num_cascades=${kf_num_cascades}
echo --kf_num_blocks=${kf_num_blocks}
echo --kf_normalize=${kf_normalize}
echo --loss_num_cascades=${loss_num_cascades}
echo --loss_num_slices=${loss_num_slices}
echo --loss_names=${loss_names}
echo --loss_weights=${loss_weights}
echo --loss_decay=${loss_decay}
echo --loss_multiscale=${loss_multiscale}
echo --enable_crop_size=${enable_crop_size}
echo --crop_target=${crop_target}
echo --optimizer=${optimizer}
echo --lr=${lr}
echo --lr_step_size=${lr_step_size}
echo --lr_gamma=${lr_gamma}
echo --weight_decay=${weight_decay}
echo --momentum=${momentum}
echo --save_keys=${save_keys}

# trainer params
mode=train
accelerator=gpu
devices=1
batch_size=1
num_workers=16
strategy=ddp
lr_step_size=40
max_epochs=50

echo --mode=${mode}
echo --accelerator=${accelerator}
echo --devices=${devices}
echo --batch_size=${batch_size}
echo --num_workers=${num_workers}
echo --strategy=${strategy}
echo --lr_step_size=${lr_step_size}
echo --max_epochs=${max_epochs}

# run script
${conda_env}/bin/python -u ${func_path} \
	--data_path=${data_path} --default_root_dir=${default_root_dir} \
	--acquisitions ${acquisitions[*]} \
	--challenge=${challenge} \
	--task=${task} \
	--train_setname=${train_setname} --val_setname=${val_setname} --test_setname=${test_setname} \
	--train_accfactor ${train_accfactor[*]} --val_accfactor ${val_accfactor[*]} --test_accfactor ${test_accfactor[*]} \
	--trainlst=${trainlst} --vallst=${vallst} --testlst=${testlst} \
	--stats_file=${stats_file} \
	--mask_type ${mask_type} --center_fractions ${center_fractions[*]} --accelerations ${accelerations[*]} \
	--scontext ${scontext} --tfcontext ${tfcontext} \
	--spatial_shrink_size ${spatial_shrink_size[*]} --scontext_shrink_size ${scontext_shrink_size[*]}  --tfcontext_shrink_size ${tfcontext_shrink_size[*]} \
	--combine_train_val ${combine_train_val} \
	--mask_correction_mode ${mask_correction_mode[*]} \
	--xt_model ${xt_model} \
	--xt_num_cascades ${xt_num_cascades} \
	--xt_chans ${xt_chans} \
	--xt_pools ${xt_pools} \
	--kt_model ${kt_model} \
	--kt_num_cascades ${kt_num_cascades} \
	--kt_num_blocks ${kt_num_blocks} \
	--kt_normalize ${kt_normalize} \
	--xf_model ${xf_model} \
	--xf_num_cascades ${xf_num_cascades} \
	--xf_chans ${xf_chans} \
	--xf_pools ${xf_pools} \
	--kf_model ${kf_model} \
	--kf_num_cascades ${kf_num_cascades} \
	--kf_num_blocks ${kf_num_blocks} \
	--kf_normalize ${kf_normalize} \
	--loss_num_cascades ${loss_num_cascades} \
	--loss_num_slices ${loss_num_slices} \
	--loss_names ${loss_names[*]} \
	--loss_weights ${loss_weights[*]} \
	--loss_decay ${loss_decay} \
	--loss_multiscale ${loss_multiscale[*]} \
	--enable_crop_size ${enable_crop_size} \
	--crop_target ${crop_target} \
	--optimizer ${optimizer} \
	--lr ${lr} \
	--lr_step_size ${lr_step_size} \
	--lr_gamma ${lr_gamma} \
	--weight_decay ${weight_decay} \
	--momentum ${momentum} \
	--save_keys ${save_keys[*]} \
	--mode=${mode} --accelerator=${accelerator} --devices=${devices} --batch_size=${batch_size} --num_workers=${num_workers} \
	--strategy=${strategy} --lr_step_size=${lr_step_size} --max_epochs=${max_epochs}
