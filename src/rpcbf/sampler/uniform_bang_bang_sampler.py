import functools as ft

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import ArrayLike, Float, PRNGKeyArray
from og.dyn_types import Disturb, HFloat, State
from og.jax_utils import jax_vmap

from rpcbf.dyn.task import Policy, Task
from rpcbf.sampler.uniform_sampler import UniformSampler


class UniformBangBangSampler(UniformSampler):
    @define
    class Cfg:
        n_samples: int
        n_samples_uni: int
        piecewise_interval_size: int

    key: PRNGKeyArray
    count: int
    task: Task = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)
    cfg: Cfg = struct.field(pytree_node=False)

    @staticmethod
    def create(key: PRNGKeyArray, task: Task, horizon: int, cfg: Cfg) -> "UniformBangBangSampler":
        count = 0
        return UniformBangBangSampler(key, count, task, horizon, cfg)

    def sample_dstb(self, key: PRNGKeyArray, n_samples, horizon, nd) -> Disturb:
        assert self.cfg.n_samples_uni <= n_samples
        assert self.cfg.n_samples_uni >= 0
        bH_dstb_uni = jr.uniform(key, minval=-1, maxval=1, shape=(self.cfg.n_samples_uni, horizon, nd))
        bang_bang_dstb = jnp.array([-1, 1])
        bH_dstb_bangbang = jr.choice(
            key, bang_bang_dstb, shape=(n_samples - self.cfg.n_samples_uni, horizon, nd), replace=True, p=None
        )
        bH_dstb = jnp.concatenate([bH_dstb_uni, bH_dstb_bangbang], axis=0)
        return bH_dstb

    def compute_h_hmax(
        self, cbf_policy: Policy, x0: State, include_h0: bool
    ) -> tuple["UniformBangBangSampler", HFloat, Float[ArrayLike, "nh H nd"], dict]:

        key = jr.fold_in(self.key, self.count)

        bH_dstb, bi_dstb = self.generate_piecewise_constant_disturbances(
            key,
            n_samples=self.cfg.n_samples,
            horizon=self.horizon,
            nd=self.task.nd,
            interval_size=self.cfg.piecewise_interval_size,
        )

        bir_dstb = ei.repeat(bi_dstb, "b i nd -> b i r nd", r=self.cfg.piecewise_interval_size)
        bH_dstb = ei.rearrange(bir_dstb, "b i r nd -> b (i r) nd")
        assert bH_dstb.shape == (self.cfg.n_samples, self.horizon, self.task.nd)

        rollout_fn = ft.partial(self.rollout, cbf_policy, x0, include_h0=include_h0)
        bh_hmax, b_info = jax.vmap(rollout_fn)(bH_dstb)
        # Note: the max for different components could come from different samples.

        assert bh_hmax.shape == (self.cfg.n_samples, self.task.nh)

        # Get the indices that achieve the max for each component.
        h_argmax = jnp.argmax(bh_hmax, axis=0)
        assert h_argmax.shape == (self.task.nh,)
        h_hmax = bh_hmax[h_argmax, jnp.arange(bh_hmax.shape[1])]

        # Get the dstb that achieves the max for each component.
        hH_dstb = bH_dstb[h_argmax]
        # assert hi_dstb.shape == (self.task.nh, self.n_intervals, self.task.nd)

        bHp1_x, bHp1h_h = b_info["Hp1_x"], b_info["Hp1h_h"]
        hHp1_x, hHp1h_h = bHp1_x[h_argmax], bHp1h_h[h_argmax]
        assert hHp1_x.shape == (self.task.nh, self.horizon + 1, self.task.nx)
        assert hHp1h_h.shape == (self.task.nh, self.horizon + 1, self.task.nh)
        # info = {"hHp1_x": hHp1_x, "hHp1h_h": hHp1h_h} | b_info
        info = {"bHp1_x": bHp1_x, "bHp1h_h": bHp1h_h, "h_hmax": h_hmax}

        new_self = self.replace(count=self.count + 1)
        return new_self, h_hmax, hH_dstb, info
