"""Custom OmegaConf resolvers used across all entry points.

Import this module once (typically via ``netcdf.config`` or ``bin/train.py``)
to register ``${slice:…}``, ``${indices:…}``, and ``${env:…}`` resolvers.

All registrations use ``replace=True`` so re-importing is safe.
"""

import os

from omegaconf import OmegaConf


def _make_slice(*args):
    args = [None if x == "None" else int(x) for x in args]
    return slice(*args)


def _make_indices(*args):
    return [int(x) for x in args]


def _make_env(key):
    return os.environ.get(key, "")


def register_resolvers():
    OmegaConf.register_new_resolver("slice", _make_slice, replace=True)
    OmegaConf.register_new_resolver("indices", _make_indices, replace=True)
    OmegaConf.register_new_resolver("env", _make_env, replace=True)


register_resolvers()
