# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.fft


COMPLEX_DIM = 2


def is_complex_data(data: torch.Tensor, complex_axis: int = -1) -> bool:
    """Returns True if data is a complex tensor at a specified dimension, i.e. complex_axis of data is of size 2,
    corresponding to real and imaginary channels..

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the check will be done. Default: -1 (last).

    Returns
    -------
    bool
        True if data is a complex tensor.
    """

    return data.size(complex_axis) == COMPLEX_DIM


def is_power_of_two(number: int) -> bool:
    """Check if input is a power of 2.

    Parameters
    ----------
    number: int

    Returns
    -------
    bool
    """
    return number != 0 and ((number & (number - 1)) == 0)


def assert_complex(data: torch.Tensor, complex_axis: int = -1, complex_last: Optional[bool] = None) -> None:
    """Assert if a tensor is complex (has complex dimension of size 2 corresponding to real and imaginary channels).

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the assertion will be done. Default: -1 (last).
    complex_last: Optional[bool]
        If true, will override complex_axis with -1 (last). Default: None.
    """
    # TODO: This is because ifft and fft or torch expect the last dimension to represent the complex axis.
    if complex_last:
        complex_axis = -1
    assert is_complex_data(
        data, complex_axis
    ), f"Complex dimension assumed to be 2 (complex valued), but not found in shape {data.shape}."


def verify_fft_dtype_possible(data: torch.Tensor, dims: Tuple[int, ...]) -> bool:
    """fft and ifft can only be performed on GPU in float16 if the shapes are powers of 2. This function verifies if
    this is the case.

    Parameters
    ----------
    data: torch.Tensor
    dims: tuple

    Returns
    -------
    bool
    """
    is_complex64 = data.dtype == torch.complex64
    is_complex32_and_power_of_two = (data.dtype == torch.float32) and all(
        is_power_of_two(_) for _ in [data.size(idx) for idx in dims]
    )

    return is_complex64 or is_complex32_and_power_of_two


def view_as_complex(data):
    """Returns a view of input as a complex tensor.

    For an input tensor of size (N, ..., 2) where the last dimension of size 2 represents the real and imaginary
    components of complex numbers, this function returns a new complex tensor of size (N, ...).

    Parameters
    ----------
    data: torch.Tensor
        Input data with torch.dtype torch.float64 and torch.float32 with complex axis (last) of dimension 2
        and of shape (N, \*, 2).

    Returns
    -------
    complex_valued_data: torch.Tensor
        Output complex-valued data of shape (N, \*) with complex torch.dtype.
    """
    return torch.view_as_complex(data)


def view_as_real(data):
    """Returns a view of data as a real tensor.

    For an input complex tensor of size (N, ...) this function returns a new real tensor of size (N, ..., 2) where the
    last dimension of size 2 represents the real and imaginary components of complex numbers.

    Parameters
    ----------
    data: torch.Tensor
        Input data with complex torch.dtype of shape (N, \*).

    Returns
    -------
    real_valued_data: torch.Tensor
        Output real-valued data of shape (N, \*, 2).
    """

    return torch.view_as_real(data)


def fft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered: bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the fft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    output_data: torch.Tensor
        The Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently fft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )

    assert_complex(data, complex_last=True)

    data = view_as_complex(data)
    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.fftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)

    data = view_as_real(data)
    return data


def ifft2(
    data: torch.Tensor,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ( 'height', 'width').
    centered: bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    output_data: torch.Tensor
        The Inverse Fast Fourier transform of the data.
    """
    if not all((_ >= 0 and isinstance(_, int)) for _ in dim):
        raise TypeError(
            f"Currently ifft2 does not support negative indexing. "
            f"Dim should contain only positive integers. Got {dim}."
        )
    assert_complex(data, complex_last=True)

    data = view_as_complex(data)
    if centered:
        data = ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    if verify_fft_dtype_possible(data, dim):
        data = torch.fft.ifftn(
            data,
            dim=dim,
            norm="ortho" if normalized else None,
        )
    else:
        raise ValueError("Currently half precision FFT is not supported.")

    if centered:
        data = fftshift(data, dim=dim)

    data = view_as_real(data)
    return data


def roll_one_dim(data: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Similar to roll but only for one dim

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: int

    Returns
    -------
    torch.Tensor
    """
    shift = shift % data.size(dim)
    if shift == 0:
        return data

    left = data.narrow(dim, 0, data.size(dim) - shift)
    right = data.narrow(dim, data.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    data: torch.Tensor,
    shift: List[int],
    dim: Union[List[int], Tuple[int, ...]],
) -> torch.Tensor:
    """Similar to numpy roll but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
    shift: tuple, int
    dim: List or tuple of ints

    Returns
    -------
    torch.Tensor
        Rolled version of data
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        data = roll_one_dim(data, s, d)

    return data


def fftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy fftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for idx in range(1, data.dim()):
            dim[idx] = idx

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for idx, dim_num in enumerate(dim):
        shift[idx] = data.shape[dim_num] // 2

    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Union[List[int], Tuple[int, ...], None] = None) -> torch.Tensor:
    """Similar to numpy ifftshift but applies to pytorch tensors.

    Parameters
    ----------
    data: torch.Tensor
        Input data.
    dim: List or tuple of ints or None
        Default: None.

    Returns
    -------
    torch.Tensor
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (data.shape[dim_num] + 1) // 2

    return roll(data, shift, dim)
