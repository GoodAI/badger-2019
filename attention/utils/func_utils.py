from typing import TypeVar, Callable, Optional

T = TypeVar('T')
R = TypeVar('R')


def apply(o: T, fn: Callable[[T], R]) -> Optional[R]:
    return None if o is None else fn(o)
