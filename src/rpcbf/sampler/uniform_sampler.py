import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
from attrs import define
from flax import struct
from jaxtyping import ArrayLike, Float, PRNGKeyArray
from og.dyn_types import Disturb, HFloat, State
from og.jax_utils import jax_vmap

from rpcbf.dyn.task import Policy, Task
from rpcbf.sampler.sampler import Sampler


class UniformSampler(struct.PyTreeNode, Sampler):
    """Uniformly samples the disturbance."""

    @define
    class Cfg:
        n_samples: int
        piecewise_interval_size: int

    key: PRNGKeyArray
    count: int
    task: Task = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)
    cfg: Cfg = struct.field(pytree_node=False)

    @staticmethod
    def create(key: PRNGKeyArray, task: Task, horizon: int, cfg: Cfg) -> "UniformSampler":
        count = 0
        return UniformSampler(key, count, task, horizon, cfg)

    def sample_dstb(self, key: PRNGKeyArray, n_samples, horizon, nd) -> Disturb:
        bH_dstb = jr.uniform(key, minval=-1, maxval=1, shape=(n_samples, horizon, nd))
        return bH_dstb

    def compute_h_hmax(
        self, cbf_policy: Policy, x0: State, include_h0: bool
    ) -> tuple["UniformSampler", HFloat, Float[ArrayLike, "nh H nd"], dict]:

        key = jr.fold_in(self.key, self.count)

        bH_dstb, bi_dstb = self.generate_piecewise_constant_disturbances(
            key, self.cfg.n_samples, self.horizon, self.task.nd, self.cfg.piecewise_interval_size
        )

        bir_dstb = ei.repeat(bi_dstb, "b i nd -> b i r nd", r=self.cfg.piecewise_interval_size)
        bH_dstb = ei.rearrange(bir_dstb, "b i r nd -> b (i r) nd")
        assert bH_dstb.shape == (self.cfg.n_samples, self.horizon, self.task.nd)

        rollout_fn = ft.partial(self.rollout, cbf_policy, x0, include_h0=include_h0)
        bh_hmax, b_info = jax.vmap(rollout_fn)(bH_dstb)
        # Note: the max for different components could come from different samples.

        # Get the indices that achieve the max for each component.
        h_argmax = jnp.argmax(bh_hmax, axis=0)
        assert h_argmax.shape == (self.task.nh,)
        h_hmax = bh_hmax[h_argmax, jnp.arange(bh_hmax.shape[1])]

        # Get the dstb that achieves the max for each component.
        hH_dstb = bH_dstb[h_argmax]
        assert hH_dstb.shape == (self.task.nh, self.horizon, self.task.nd)

        bHp1_x, bHp1h_h = b_info["Hp1_x"], b_info["Hp1h_h"]
        hHp1_x, hHp1h_h = bHp1_x[h_argmax], bHp1h_h[h_argmax]
        assert hHp1_x.shape == (self.task.nh, self.horizon + 1, self.task.nx)
        assert hHp1h_h.shape == (self.task.nh, self.horizon + 1, self.task.nh)
        info = {"hHp1_x": hHp1_x, "hHp1h_h": hHp1h_h, "h_hmax": h_hmax}

        new_self = self.replace(count=self.count + 1)
        return new_self, h_hmax, hH_dstb, info

    def compute_h_hmax_from_dstb(
        self, cbf_policy: Policy, x0: State, hi_dstb: Float[ArrayLike, "nh ni nd"], include_h0: bool
    ) -> HFloat:
        bh_hmax, info = jax.vmap(ft.partial(self.rollout, cbf_policy, x0, include_h0=include_h0))(hi_dstb)
        h_hmax = jnp.max(bh_hmax, axis=0)
        return h_hmax

    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value(self, cbf_policy: Policy, x0: State, include_h0: bool = False):
        new_self, h_hmax, hH_dstb, info = self.compute_h_hmax(cbf_policy, x0, include_h0)
        return new_self, h_hmax, info



    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value_and_grad(self, cbf_policy: Policy, x0: State, include_h0: bool = False):
        def compute_value(x0_):
            return self.compute_h_hmax_from_dstb(cbf_policy, x0_, hH_dstb, include_h0)

        new_self, h_hmax, hH_dstb, info = self.compute_h_hmax(cbf_policy, x0, include_h0)
        assert h_hmax.shape == (self.task.nh,)

        # compute_value_fn = ft.partial(compute_value, hT_dstb)
        grad_h_hmax = jax.jacobian(compute_value)(x0)
        assert grad_h_hmax.shape == (self.task.nh, self.task.nx)

        h0_dstb = hH_dstb[:, 0]
        f_fn = ft.partial(self.task.f, x0)
        h_f = jax_vmap(f_fn)(h0_dstb)
        G_fn = ft.partial(self.task.G, x0)
        h_G = jax_vmap(G_fn)(h0_dstb)
        info = {"hx_gradhmax": grad_h_hmax} | info

        return new_self, hH_dstb, h_hmax, grad_h_hmax, h_f, h_G, info

    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value_and_hess(self, cbf_policy: Policy, x0: State, include_h0: bool = False):
        def compute_value(x0_):
            return self.compute_h_hmax_from_dstb(cbf_policy, x0_, hH_dstb, include_h0)

        new_self, h_hmax, hH_dstb = self.compute_h_hmax(cbf_policy, x0, include_h0)
        assert h_hmax.shape == (self.task.nh,)

        def compute_value_hh(hh):
            def fn(x0_):
                return compute_value(x0_)[hh]

            return fn

        def get_jachess(hh):
            fn = compute_value_hh(hh)
            grad = jax.grad(fn)
            hess = jax.hessian(fn)
            return grad, hess

        h_Vx, h_Vxx = jax.vmap(get_jachess)(jnp.arange(self.task.nh))
        assert h_Vx.shape == (self.task.nh, self.task.nx)
        assert h_Vxx.shape == (self.task.nh, self.task.nx, self.task.nx)

        h0_dstb = hH_dstb[:, 0]
        f_fn = ft.partial(self.task.f, x0)
        h_f = jax_vmap(f_fn)(h0_dstb)
        G_fn = ft.partial(self.task.G, x0)
        h_G = jax_vmap(G_fn)(h0_dstb)

        return new_self, h0_dstb.squeeze(), h_hmax, h_Vx, h_Vxx, h_f, h_G
