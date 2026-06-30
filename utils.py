import time
import functools
from typing import Callable, List, Tuple, Optional, Any

import tsp


def timeit(func: Callable[..., Any]) -> Any:
    """Decorator that records the elapsed time of a function call.

    The decorated function will have a `.last_duration` attribute with the
    most recent run time in seconds.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        wrapper.last_duration = t1 - t0  
        return res

    wrapper.last_duration = 0.0  
    return wrapper


def log_calls(func: Callable[..., Any]) -> Any:
    """Decorator that logs each call (args, kwargs) to an attribute `.call_log`.

    Useful for tests and lightweight tracing.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.call_log.append((args, kwargs))  
        return func(*args, **kwargs)

    wrapper.call_log = []  
    return wrapper


def make_simulated_annealing(
    T0: float = 1.0, alpha: float = 0.995, steps: int = 10000
) -> Callable:
    """Closure factory that returns a configured simulated annealing solver.

    Returned function signature: f(dist, initial_tour=None) -> list
    """

    def sa_solver(dist: Any, initial_tour: Optional[List[int]] = None) -> List[int]:
        return tsp.simulated_annealing(
            dist,
            initial_tour=initial_tour,
            T0=T0,
            alpha=alpha,
            steps=steps,
        )

    return sa_solver


def distance_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D points.

    >>> distance_between((0,0), (3,4))
    5.0
    >>> distance_between((1.5, 2.5), (1.5, 2.5))
    0.0
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5
