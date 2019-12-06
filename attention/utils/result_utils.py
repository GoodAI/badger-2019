from collections import defaultdict, Hashable
from typing import Iterable, Dict, Any, Callable, List, Set, Tuple

DictStr = Dict[str, Any]


def dict_to_tuple(data: Dict[str, Any]) -> Tuple[Tuple[str, Any]]:
    # noinspection PyTypeChecker
    return tuple(sorted(data.items()))


def tuple_to_dict(data: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in data}


def dict_hashable_values(items: Dict[str, Any]) -> Dict[str, Hashable]:
    """Return new dictionary where all values are Hashable (value is converted to string when not Hashable)"""
    return {k: v if isinstance(v, Hashable) else str(v) for k, v in items.items()}


def dict_omit_keys(items: Dict[str, Any], omit_keys: Set[str]) -> Dict[str, Any]:
    """Return new dictionary where provided keys are omitted"""
    return {k: v for k, v in items.items() if k not in omit_keys}


def find_common_keys(items: Iterable[DictStr], key_extractor: Callable[[DictStr], DictStr]) -> Dict[str, Any]:
    """
    Find common keys in keys - i.e. common config parameters
    Args:
        items:
        key_extractor:

    Returns:

    """
    keys_usages = defaultdict(int)
    total = 0
    for item in items:
        key = dict_to_tuple(dict_hashable_values(key_extractor(item)))
        for k in key:
            keys_usages[k] += 1
        total += 1
    return tuple_to_dict([k for k, v in keys_usages.items() if v == total])


def group_dicts(items: Iterable[DictStr], key_extractor: Callable[[DictStr], DictStr],
                value_extractor: Callable[[DictStr], Any] = None) -> Dict[List[Tuple[str, Any]], Any]:
    """
    Group results by config - groups experiments by config
    Args:
        items: Items to be grouped, i.e. sacred runs
        key_extractor: function to read key (config) from the item
        value_extractor: function to return value to be appended to list (i.e. run_id or whole run)

    Returns:
        Dictionary where keys are unique configs, values are result of value extractor (i.e. run_id)
    """
    result = defaultdict(list)
    for item in items:
        key = dict_to_tuple(dict_hashable_values(key_extractor(item)))
        result[key].append(value_extractor(item))
    return result
