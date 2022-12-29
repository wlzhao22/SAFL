import numpy as np
import torch 
from torch import Tensor


def pad_ones_np(array, dim=0):
    shape = list(array.shape)
    shape[dim] = 1
    return np.concatenate((array, np.ones(shape, array.dtype)), axis=dim)

def pad_ones_torch(tensor, dim=0):
    shape = list(tensor.shape) 
    shape[dim] = 1
    return torch.cat((tensor, tensor.new_ones(shape)), dim=dim)

def pad_ones(arr, dim=0):
    if isinstance(arr, Tensor):
        return pad_ones_torch(arr, dim)
    return pad_ones_np(arr, dim)

def pad_zeros_np(array, dim=0):
    shape = list(array.shape) 
    shape[dim] = 1 
    return np.concatenate((array, np.zeros(shape, array.dtype)), axis=dim)


def pad_zeros_torch(tensor, dim=0):
    shape = list(tensor.shape) 
    shape[dim] = 1
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=dim)


def pad_zeros(arr, dim=0):
    if isinstance(arr, Tensor):
        return pad_zeros_torch(arr, dim)
    return pad_zeros_np(arr, dim)

