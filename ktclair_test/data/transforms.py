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

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import fastmri
from fastmri.data.transforms import to_tensor
from einops import rearrange


def mask_correction(
	data: torch.Tensor, mode: Sequence[str] = ['sy',], eps=1e-11
):
	data_shape = tuple(data.shape) # (tf sz sc sy sx 2)
	mask_shape = (1,) * len(data_shape[:-3]) + data_shape[-3:-1] + (1,) # (1 1 1 sy sx 1)
	mask = torch.ones(mask_shape)

	if mode is not None:
		# fix sx
		if 'sx' in mode:
			data1d_sx = fastmri.complex_abs(data).mean(dim=(0,1,2,3))
			cent = data_shape[4] // 2
			left = torch.diff(data1d_sx[:cent])
			right = torch.diff(data1d_sx[-cent:].flip([0]))
			diff = torch.abs(left - right)
			# detect peak in the range of [3, -cent//3]
			diff[:3] = 0
			diff[-cent//3:] = 0
			# peak
			offset = torch.argmax(diff) + 1
			left = data1d_sx[:offset].mean()
			right = data1d_sx[-offset:].mean()
			# mask (1 1 1 sy sx 1)
			if left < right:
				if right / left > 3:
					mask[...,:offset,:] = 0
			else:
				if left / right > 3:
					mask[...,-offset:,:] = 0
		# fix sy
		if 'sy' in mode:
			data1d_sy = fastmri.complex_abs(data).mean(dim=(0,1,2,4))
			data1d_sy = data1d_sy > eps
			data1d_sy = data1d_sy.nonzero(as_tuple=True)[0]
			left = data1d_sy[0]
			right = data1d_sy[-1]
			# mask (1 1 1 sy sx 1)
			mask[...,:left,:,:] = 0
			mask[...,right+1:,:,:] = 0
	else:
		pass

	return mask


def apply_mask(
	data: torch.Tensor,
	offset: Optional[int] = None,
	padding: Optional[Sequence[int]] = None,
	acq_shape: Tuple[int] = None,
	orig_mask: Dict[str, np.ndarray] = None,
	eps: float = 1e-11,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
	"""
	Subsample given k-space by multiplying with a mask.

	Args:
		data: The input k-space data. This should have at least 3 dimensions,
			where dimensions -3 and -2 are the spatial dimensions, and the
			final dimension has size 2 (for complex values).
		padding: Padding value to apply for mask.

	Returns:
		tuple containing:
			masked data: Subsampled k-space data.
			mask: The generated mask.
			num_low_frequencies: The number of low-resolution frequency samples
				in the mask.
	"""
	data_shape = tuple(data.shape) # (tf sz sc sy sx 2)
	if acq_shape is None:
		acq_shape = data_shape[-3:-1]
	else:
		assert len(acq_shape) == 2

	mask_shape = (1,) * len(data_shape[:-3]) + acq_shape + (1,) # (1 1 1 sy sx 1)

	# load/generate mask
	if orig_mask is not None:
		# load mask
		acc = [k for k in orig_mask.keys()][0]
		mask = orig_mask[acc] # (sy sx)
	else:
		# generate mask along the dimensions -2
		mask = torch.ones(acq_shape) # (sy sx)
		data1d_sy = fastmri.complex_abs(data).mean(dim=(0,1,2,4)) # (sy)
		mask[data1d_sy < eps,:] = 0.0
		mask = mask.numpy() # (sy sx)

	# get low frequency line locations
	mask1d = mask.mean(axis=1) > 0.9
	cent = mask1d.shape[0] // 2
	left = np.argmin(np.flip(mask1d[:cent]))
	right = np.argmin(mask1d[cent:])
	num_low_frequencies = left + right
	# low_freq_left_ind = cent - left
	mask = torch.from_numpy(mask).to(data).reshape(mask_shape)
	# num_low_frequencies = 24

	masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

	return masked_data, mask, num_low_frequencies


def create_additional_mask(
	data_shape: Union[int, Tuple[int]],
	num_low_frequencies: int,
	spatial_shrink_size: Union[int, Tuple[int]] = None,
	tfcontext_shrink_size: Union[int, Tuple[int]] = None,
	scontext_shrink_size: Union[int, Tuple[int]] = None,
) -> torch.Tensor:
	"""
	Creates a boolean mask.

	Args:
		shape: shape of the desired mask
		num_low_frequencies: Variable telling number of lines (1D)
			or radius (2D) to be sampled in the centre.
		offset: offset for radius/line reduction

	Returns:
		a boolean mask.
	"""
	data_shape = tuple(data_shape) # (tf sz sc sy sx 2)
	tf, sz, _, _, _, _ = data_shape
	mask_shape = (1,) * len(data_shape[:-3]) + data_shape[-3:-1] + (1,) # (1 1 1 sx sy 1)

	spatial_shrink_size = spatial_shrink_size if spatial_shrink_size is not None else 0
	tfcontext_shrink_size = tfcontext_shrink_size if tfcontext_shrink_size is not None else 0
	scontext_shrink_size = scontext_shrink_size if scontext_shrink_size is not None else 0
	spatial_shrink_size = [spatial_shrink_size] if not isinstance(spatial_shrink_size, (list, tuple)) else spatial_shrink_size
	tfcontext_shrink_size = [tfcontext_shrink_size] if not isinstance(tfcontext_shrink_size, (list, tuple)) else tfcontext_shrink_size
	scontext_shrink_size = [scontext_shrink_size] if not isinstance(scontext_shrink_size, (list, tuple)) else scontext_shrink_size
	if not (len(spatial_shrink_size) == len(tfcontext_shrink_size) == len(scontext_shrink_size)):
		raise

	additional_masks = torch.stack([torch.stack([
		center_rectangle_mask(mask_shape[-3:-1], sampled_shape, offset).reshape(mask_shape) for (
		sampled_shape, offset) in zip(
			((num_low_frequencies, mask_shape[-2]), (num_low_frequencies, mask_shape[-2]), mask_shape[-3:-1]),
			(0, ssize, ssize),
		)
	], dim=0) for ssize in spatial_shrink_size], dim=1) # (3 s 1 1 1 sy sx 1)

	if tf > 1 or sz > 1:
		additional_masks = additional_masks.repeat(1, 1, tf, sz, 1, 1, 1, 1) # (3 s tf sz 1 sy sx 1)
		for i, ssize in enumerate(tfcontext_shrink_size):
			if ssize > 0 and tf > ssize:
				ssize = min(tf // 2, ssize)
				additional_masks[1:, i, 0:ssize] = 0.0
				additional_masks[1:, i, -ssize:] = 0.0
		for i, ssize in enumerate(scontext_shrink_size):
			if ssize > 0 and sz > ssize:
				ssize = min(sz // 2, ssize)
				additional_masks[1:, i, :, 0:ssize] = 0.0
				additional_masks[1:, i, :, -ssize:] = 0.0

	return additional_masks


def center_rectangle_mask(
	shape: Union[int, Tuple[int]], length: Union[int, Tuple[int]], offset: Union[int, Tuple[int]] = 0
) -> torch.Tensor:
	"""
	Creates a boolean mask with centered rectangle using a pre-defined length.

	Args:
		shape: shape of the desired mask
		length: length of the desired rectangle
		offset: offset for rectangle reduction

	Returns:
		a boolean mask with centered rectangle using a pre-defined length.
	"""
	if not isinstance(shape, (tuple, list)):
		shape = [shape] * 2
	if not isinstance(length, (tuple, list)):
		length = [length] * 2
	if not isinstance(offset, (tuple, list)):
		offset = [offset] * 2
	assert len(shape) == len(length) == len(offset)

	pad_x = (shape[0] - length[0]) // 2 # favor left when even lines
	pad_y = (shape[1] - length[1]) // 2 # favor left when even lines

	pad_x += offset[0]
	pad_y += offset[1]

	mask = torch.zeros(shape, dtype=torch.bool)
	mask[
		pad_x : pad_x + length[0] - offset[0] * 2,
		pad_y : pad_y + length[1] - offset[1] * 2
	] = True

	return mask


class CMRSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        kspace: Input k-space of shape (num_coils, rows, cols, 2) for multi-coil data.
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    tframe_num: int
    slice_num: int
    max_value: float
    additional_masks: torch.Tensor


class CMRDataTransform:
	"""
	Data Transformer for training VarNet models.
	"""

	def __init__(
		self,
		spatial_shrink_size: Optional[Union[int, Tuple[int]]] = None,
		tfcontext_shrink_size: Optional[Union[int, Tuple[int]]] = None,
		scontext_shrink_size: Optional[Union[int, Tuple[int]]] = None,
		mask_correction_mode: Optional[Sequence[str]] = None,
	):
		self.spatial_shrink_size = spatial_shrink_size
		self.tfcontext_shrink_size = tfcontext_shrink_size
		self.scontext_shrink_size = scontext_shrink_size
		self.mask_correction_mode = mask_correction_mode

	def __call__(
		self,
		kspace: np.ndarray,
		mask: np.ndarray,
		target: Optional[np.ndarray],
		attrs: Dict,
		fname: str,
		tframe_num: int,
		slice_num: int,
	) -> CMRSample:
		"""
		Args:
			kspace: Input k-space of shape (num_coils, rows, cols) for multi-coil data.
			mask: Mask from the test dataset.
			target: Target image.
			attrs: Acquisition related information stored in the HDF5 object.
			fname: File name.
			tframe_num: time frame number
			slice_num: Serial number of the slice.

		Returns:
			A CMRSample with the masked k-space, sampling mask, target
			image, the filename, the slice number, the maximum image value
			(from target), the target crop size, and the number of low
			frequency lines sampled.
		"""
		### kspace_torch
		if len(kspace.shape) == 4: # singlecoil (tf sz sy sx)
			kspace = kspace[:,:,None]
		if len(kspace.shape) != 5: # multicoil (tf sz sc sy sx)
			raise ValueError(f"kspace should have 5D dimensions. Got {len(kspace.shape)} dimensions instead.")
		tf, sz, sc, sy, sx = kspace.shape
		kspace_torch = to_tensor(kspace) # (tf sz sc sy sx 2)

		### target_torch
		if target is not None:
			target_torch = to_tensor(target) # (tf sz sy sx)
			max_value = attrs["max"] if "max" in attrs.keys() else target_torch.max()
		else:
			target_torch = None
			max_value = 0.0

		### apply mask
		masked_kspace, mask_torch, num_low_frequencies = apply_mask(
			kspace_torch, orig_mask=mask,
		) # (tf sz sc sy sx 2), (1 1 1 sy sx 1)

		""" additional masks (3 s tf sz 1 sy sx 1)
		center_mask
		shrinked_center_mask
		shrinked_region_mask
		"""
		additional_masks = create_additional_mask(
			masked_kspace.shape,
			num_low_frequencies,
			self.spatial_shrink_size,
			self.tfcontext_shrink_size,
			self.scontext_shrink_size,
		) # (3 s tf sz 1 sy sx 1)

		### mask_correct
		if self.mask_correction_mode is not None:
			mask_correct = mask_correction(kspace_torch, mode=self.mask_correction_mode) # (1 1 1 sy sx 1)
			mask_torch = mask_torch * mask_correct
			additional_masks[0] = additional_masks[0] * mask_correct[None] # center_mask
			additional_masks[1] = additional_masks[1] * mask_correct[None] # shrinked_center_mask
		
		### 2D/3D
		if tf > 1 and sz > 1:
			raise ValueError("exploring both tf and sz contexts is not supported!")
		if tf == 1 and sz == 1:
			# 2D
			masked_kspace = masked_kspace[tf//2,sz//2] # (sc sy sx 2)
			mask_torch = mask_torch[tf//2,sz//2] # (1 sy sx 1)
			additional_masks = additional_masks[:,:,tf//2,sz//2] # (3 s 1 sy sx 1)
			target_torch = target_torch[tf//2,sz//2] if target_torch is not None else None # (sy sx)
		elif tf > 1:
			# 3D with tf, converting (tf sz sc sy sx 2) to (sz sc tf sy sx 2)
			masked_kspace = rearrange(masked_kspace, 'tf sz sc sy sx comp -> sz sc tf sy sx comp')
			mask_torch = rearrange(mask_torch, 'tf sz sc sy sx comp -> sz sc tf sy sx comp')
			additional_masks  = rearrange(additional_masks, 'three s tf sz sc sy sx comp -> three s sz sc tf sy sx comp')
			target_torch = rearrange(target_torch, 'tf sz sy sx -> sz tf sy sx') if target_torch is not None else None
			masked_kspace = masked_kspace[sz//2] # (sc tf sy sx 2)
			mask_torch = mask_torch[sz//2] # (1 1 sy sx 1)
			additional_masks = additional_masks[:,:,sz//2] # (3 s 1 tf sy sx 1)
			target_torch = target_torch[sz//2] if target_torch is not None else None # (tf sy sx)
		elif sz > 1:
			# 3D with sz, converting (tf sz sc sy sx 2) to (tf sc sz sy sx 2)
			masked_kspace = rearrange(masked_kspace, 'tf sz sc sy sx comp -> tf sc sz sy sx comp')
			mask_torch = rearrange(mask_torch, 'tf sz sc sy sx comp -> tf sc sz sy sx comp')
			additional_masks  = rearrange(additional_masks, 'three s tf sz sc sy sx comp -> three s tf sc sz sy sx comp')
			target_torch = rearrange(target_torch, 'tf sz sy sx -> tf sz sy sx') if target_torch is not None else None
			masked_kspace = masked_kspace[tf//2] # (sc sz sy sx 2)
			mask_torch = mask_torch[tf//2] # (1 1 sy sx 1)
			additional_masks = additional_masks[:,:,tf//2] # (3 s 1 sz sy sx 1)
			target_torch = target_torch[tf//2] if target_torch is not None else None # (sz sy sx)
		else:
			raise

		### CMRSample
		sample = CMRSample(
			masked_kspace=masked_kspace,
			mask=mask_torch.to(torch.bool),
			num_low_frequencies=num_low_frequencies,
			target=target_torch if target_torch is not None else torch.tensor(0),
			fname=fname,
			tframe_num=tframe_num,
			slice_num=slice_num,
			max_value=max_value,
			additional_masks=additional_masks.to(torch.bool),
		)

		return sample
