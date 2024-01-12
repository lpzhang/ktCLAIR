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

import ast
import functools
import importlib
from typing import Callable
import torch
from collections import OrderedDict


def str_to_class(module_name: str, function_name: str) -> Callable:
    """Convert a string to a class Base on: https://stackoverflow.com/a/1176180/576363.

    Also support function arguments, e.g. ifft(dim=2) will be parsed as a partial and return ifft where dim has been
    set to 2.


    Examples
    --------
    >>> def mult(f, mul=2):
    >>>    return f*mul

    >>> str_to_class(".", "mult(mul=4)")
    >>> str_to_class(".", "mult(mul=4)")
    will return a function which multiplies the input times 4, while

    >>> str_to_class(".", "mult")
    just returns the function itself.

    Parameters
    ----------
    module_name: str
        e.g. direct.data.transforms
    function_name: str
        e.g. Identity
    Returns
    -------
    object
    """
    tree = ast.parse(function_name)
    func_call = tree.body[0].value  # type: ignore
    args = [ast.literal_eval(arg) for arg in func_call.args] if hasattr(func_call, "args") else []
    kwargs = (
        {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords} if hasattr(func_call, "keywords") else {}
    )

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)

    if not args and not kwargs:
        return getattr(module, function_name)
    return functools.partial(getattr(module, func_call.func.id), *args, **kwargs)


def save_state_dict_from_checkpoint(checkpoint_path: str, state_dict_path: str, net_startswith: str ='model.'):
    state_dict = torch.load(checkpoint_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(net_startswith):
            new_state_dict[k[len(net_startswith):]] = v
    torch.save(new_state_dict, state_dict_path)
    print("state_dict saved in:", state_dict_path)


def state_dict_parser(checkpoint_path: str, net_startswith: str ='model.'):
    state_dict = torch.load(checkpoint_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(net_startswith):
            new_state_dict[k[len(net_startswith):]] = v
    return new_state_dict
