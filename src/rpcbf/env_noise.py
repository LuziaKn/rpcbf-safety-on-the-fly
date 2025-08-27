from typing import Protocol

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import PRNGKeyArray
from og.dyn_types import Disturb

from rpcbf.dyn.task import Task


class EnvNoise(Protocol):

    @property

    def sample_dstb(self, key_base: PRNGKeyArray, x: np.ndarray, kk: int) -> Disturb:
        """The key should be the same for the entire trajectory.

        To get a new key, use jr.fold_in(key_base, kk)."""
        ...

    def name(self):
        ...

class UniformEnvNoise(EnvNoise):
    def __init__(self, task: Task):
        self.task = task

    def sample_dstb(self, key_base: PRNGKeyArray, x: np.ndarray, kk: int):
        key_step = jr.fold_in(key_base, kk)
        return jr.uniform(key_step, shape=(self.task.nd,), minval=-1, maxval=1)

    def name(self):
        return self.__class__.__name__


class ZeroEnvNoise(EnvNoise):
    def __init__(self, task: Task):
        self.task = task

    def sample_dstb(self, key_base: PRNGKeyArray, x: np.ndarray, kk: int):
        return jnp.zeros(self.task.nd)

    def name(self):
        return self.__class__.__name__


class PiecewiseBangBangEnvNoise(EnvNoise):
    """Implement the piecewise interval noise by making sure the key is the same within the interval."""

    def __init__(self, task: Task, piecewise_interval_size: int):
        self.task = task
        self.piecewise_interval_size = piecewise_interval_size

    def sample_dstb(self, key_base: PRNGKeyArray, x: np.ndarray, kk: int):
        interval_idx = kk // self.piecewise_interval_size
        key_step = jr.fold_in(key_base, interval_idx)
        return jr.choice(key_step, shape=(self.task.nd,), a=jnp.array([-1, 1]))

    def name(self):
        return self.__class__.__name__
