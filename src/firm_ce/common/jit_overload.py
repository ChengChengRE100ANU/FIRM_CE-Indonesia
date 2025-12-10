"""
Numba functions are overloaded to allow for JIT to be switched off,
allowing for debugging with the Python interpreter instead.
"""

from firm_ce.common.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import njit, prange
    from numba.experimental import jitclass
else:

    def jitclass(spec):
        def decorator(cls):
            return cls

        return decorator

    def njit(func=None, **kwargs):
        if func is not None:
            return func

        def wrapper(f):
            return f

        return wrapper

    def prange(x):
        return range(x)
