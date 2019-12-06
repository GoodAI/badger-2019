from _operator import mul
from functools import reduce
from typing import Union, List

import gym
import numpy as np


def get_total_space_size(spaces: Union[gym.spaces.Box, gym.spaces.Tuple]) -> int:
    """ Return the total dimensionality of the gym.spaces.Box or multiple boxes
    """
    if isinstance(spaces, gym.spaces.Box):
        return get_gym_box_dimensionality(spaces)
    elif isinstance(spaces, gym.spaces.Tuple):
        return sum([get_gym_box_dimensionality(space) for space in spaces])
    elif isinstance(spaces, list):
        return sum([get_gym_box_dimensionality(space) for space in spaces])
    raise Exception('unexpected spaces type, expected Box or Tuple of Boxes')


def get_gym_box_dimensionality(space: gym.spaces.Box) -> int:
    """ Return the dimensionality of the gym.spaces.Box (continuous observations/actions)
    """
    if len(space.shape) == 0:
        return 1
    return reduce(mul, space.shape, 1)


def concatenate_spaces(spaces: List[gym.Space]) -> gym.Space:
    """Concatenate list of gym.Spaces to one global gym space"""

    for space in spaces:
        assert isinstance(space, gym.spaces.Box), 'concatenation of boxes supported only'

    lows = np.array([space.low for space in spaces])
    highs = np.array([space.high for space in spaces])

    result = gym.spaces.Box(low=lows, high=highs)
    return result