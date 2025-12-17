# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch


def swap(a, b):
    if type(a) == torch.Tensor:
        return b.clone(), a.clone()
    else:
        raise NotImplementedError


def swap_ref(a, b):
    if type(a) == torch.Tensor:
        temp = a.clone()
        a[:] = b.clone()
        b[:] = temp
    else:
        raise NotImplementedError
