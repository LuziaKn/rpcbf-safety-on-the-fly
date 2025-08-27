import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import numpy as np
from attrs import define
from flax import struct
from jaxtyping import PRNGKeyArray
from og.dyn_types import State

from rpcbf.dyn.task import Policy, Task
from rpcbf.sampler.sampler import Sampler


class ZeroSampler(struct.PyTreeNode, Sampler):
    """Only one sample of disturbance = 0."""

    @define
    class Cfg:
        piecewise_interval_size=1

    task: Task = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)
    cfg: Cfg = struct.field(pytree_node=False)

    @staticmethod
    def create(key: PRNGKeyArray, task: Task, horizon: int, cfg: Cfg) -> "ZeroSampler":
        return ZeroSampler(task, horizon, cfg)

    def compute_h_hmax(self, policy: Policy, x0: State, include_h0: bool):
        T_dstb = jnp.zeros((self.horizon, self.task.nd))
        h_hmax, info = self.rollout(policy, x0, T_dstb, include_h0=include_h0)
        return h_hmax, info

    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value(self, policy: Policy, x0: State, include_h0: bool = False):
        h_hmax, info = self.compute_h_hmax(policy, x0, include_h0)
        return self, h_hmax, info

    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value_and_grad(self, policy: Policy, x0: State, include_h0: bool = False):
        def compute_value(x0_):
            h_hmax, _ = self.compute_h_hmax(policy, x0_, include_h0)
            return h_hmax

        h_hmax, info = self.compute_h_hmax(policy, x0, include_h0)
        assert h_hmax.shape == (self.task.nh,)
        if not include_h0:
            h_hmax = jnp.maximum(h_hmax, self.task.h_vec(x0))

        grad_h_hmax = jax.jacobian(compute_value)(x0)
        assert grad_h_hmax.shape == (self.task.nh, self.task.nx)

        hT_dstb = np.zeros((self.task.nh, self.horizon, self.task.nd))
        dstb_zero = np.zeros(self.task.nd)
        h_f = ei.repeat(self.task.f(x0, dstb_zero), "nx -> nh nx", nh=self.task.nh)
        h_G = ei.repeat(self.task.G(x0, dstb_zero), "nx nu -> nh nx nu", nh=self.task.nh)

        info = {"hx_gradhmax": grad_h_hmax, "h_f": h_f, "h_G": h_G} | info
        return self, hT_dstb, h_hmax, grad_h_hmax, h_f, h_G, info

    @ft.partial(jax.jit, static_argnames=("include_h0",))
    def get_value_and_hess(self, policy: Policy, x0: State, u_nom, include_h0: bool = False):
        def compute_value(x0_):
            h_hmax, _ = self.compute_h_hmax(policy, x0_, include_h0)
            return h_hmax

        h_hmax, info = self.compute_h_hmax(policy, x0, include_h0)
        assert h_hmax.shape == (self.task.nh,)

        def compute_value_hh(hh):
            def fn(x0_):
                return compute_value(x0_)[hh]

            return fn

        def get_jachess(hh):
            fn = compute_value_hh(hh)
            grad = jax.grad(fn)(x0)
            hess = jax.hessian(fn)(x0)
            return grad, hess

        h_Vx, h_Vxx = jax.vmap(get_jachess)(jnp.arange(self.task.nh))
        assert h_Vx.shape == (self.task.nh, self.task.nx)
        assert h_Vxx.shape == (self.task.nh, self.task.nx, self.task.nx)

        dstb = np.zeros(self.task.nd)
        h_f = ei.repeat(self.task.f(x0, dstb), "nx -> nh nx", nh=self.task.nh)
        h_G = ei.repeat(self.task.G(x0, dstb), "nx nu -> nh nx nu", nh=self.task.nh)

        def get_Vdot(x0_, Vx):
            f = self.task.f(x0_, dstb)
            G = self.task.G(x0_, dstb)
            xdot_nom = f + G @ u_nom
            return jnp.dot(Vx, xdot_nom)

        def get_h_Vdot(x0_):
            h_Vx_ = jax.jacobian(compute_value)(x0_)
            return jax.vmap(ft.partial(get_Vdot, x0_))(h_Vx_)

        hx_gradVdot = jax.jacobian(get_h_Vdot)(x0)

        return self, dstb, h_hmax, h_Vx, h_Vxx, hx_gradVdot, h_f, h_G
