from argparse import Namespace

import torch

"""Determine the device automatically by default"""
device_used = 'cuda' if torch.cuda.is_available() else 'cpu'


def choose_device(config: Namespace):
    """Possibility to force_cpu by the experiment config"""
    if config.force_cpu:
        global device_used
        device_used = 'cpu'


def my_device() -> str:
    """Returns the device that should be used in this experiment"""
    return device_used

