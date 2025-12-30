import argparse
import glob
import itertools
import os
import pickle
import pandas as pd
import numpy as np
from math import floor
import torch
from collections.abc import Iterable, Callable
from numpy.typing import ArrayLike
from scipy import linalg

def makeGrid(pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

def str_or_none(v):
    if isinstance(v, str):
        if v.casefold() == 'none':
            return None
        else:
            return v
    elif v is None:
        return None
    else:
        raise argparse.ArgumentTypeError('Pass a string or None')

def str2listints(v):
    if (v is None) or isinstance(v, list):
        return v
    elif v=="None":
        return None
    elif isinstance(v, str):
        v = v[1:-1].split(',')
        v = [float(i) for i in v]
        return v
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.casefold() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.casefold() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if (v is None) or isinstance(v, list):
        return v
    elif v=="None":
        return None
    else:
        v = v.split('\'')
        v = [i for n, i in enumerate(v) if n % 2 != 0]
        return v

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0)" % (x,))
    return x

def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int_nozero(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def findSeq(inputList):
    """
    Taken from https://stackoverflow.com/questions/38621615/find-the-set-of-elements-
    in-a-list-without-sorting-in-python
    """
    dictionary = {}
    newList = []
    for elem in inputList:
        if elem not in dictionary:
            dictionary[elem] = True
            newList += [elem]
    return newList

def get_aug_idx(augmentation_to_idx):
    i = augmentation_to_idx
    if i == 'flip_horizontal':
        return 0
    elif i == 'flip_vertical':
        return 1
    elif i == 'add_band_noise':
        return 2
    elif i == 'add_eeg_artifact':
        return 3
    elif i == 'add_noise_snr':
        return 4
    elif i == 'channel_dropout':
        return 5
    elif i == 'masking':
        return 6
    elif i == 'warp_signal':
        return 7
    elif i == 'random_FT_phase':
        return 8
    elif i == 'phase_swap':
        return 9

def get_aug_name(idx_to_augmentation):
    i = idx_to_augmentation
    if i == 0:
        return "None"
    elif i == 1:
        return 'flip_horizontal'
    elif i == 2:
        return 'flip_vertical'
    elif i == 3:
        return 'add_band_noise'
    elif i == 4:
        return 'add_eeg_artifact'
    elif i == 5:
        return 'add_noise_snr'
    elif i == 6:
        return 'channel_dropout'
    elif i == 7:
        return 'masking'
    elif i == 8:
        return 'warp_signal'
    elif i == 9:
        return 'random_FT_phase'
    elif i == 10:
        return 'phase_swap'