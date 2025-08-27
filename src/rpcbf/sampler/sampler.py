import functools as ft
from typing import Literal, get_args

import ipdb
import jax
from jax import lax as lax
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from loguru import logger
from og.dyn_types import Control, Disturb, State
from og.jax_utils import jax_vmap

from rpcbf.dyn.task import Policy, Task
from rpcbf.interp import max_cubic_spl_fast

MaxType = Literal["discrete", "discrete_dense", "cubic_spline"]


class Sampler:
    task: Task

    def rollout(
        self,
        cbf_policy: Policy,
        x0: State,
        H_dstb: Disturb,
        include_h0: bool,
        max_type: MaxType = "cubic_spline",
    ):
        # TODO: Terminal value function
        def body(carry, dstb):
            x = carry
            x_new = self.task.step_with_policy(x, cbf_policy, dstb)
            return x_new, x_new

        assert max_type in get_args(MaxType)

        h_hmax0 = self.task.h_vec(x0)

        carry_init = x0
        x_final, H_xnew = lax.scan(body, carry_init, H_dstb)

        Hh_hnew = jax.vmap(self.task.h_vec)(H_xnew)
        Hp1_x = jnp.concatenate([x0[None], H_xnew], axis=0)
        Hp1h_h = jnp.concatenate([h_hmax0[None], Hh_hnew], axis=0)

        if max_type == "discrete":
            h_hmax = jnp.max(Hh_hnew, axis=0)
        elif max_type == "cubic_spline":
            Hp1_t = jnp.linspace(0, 1, Hp1h_h.shape[0])
            h_hmax, _ = jax.vmap(ft.partial(max_cubic_spl_fast, Hp1_t, include_initial=True), in_axes=1)(Hp1h_h)
        else:
            raise NotImplementedError("")

        info = {"Hp1_x": Hp1_x, "Hp1h_h": Hp1h_h, "h_hmax": h_hmax}

        return h_hmax, info

    def sample_dstb(self, key: PRNGKeyArray, n_samples, horizon, nd) -> Disturb:
        ...

    def get_value(self, cbf_policy: Policy, x0: State, include_h0: bool = False):
        ...

    def get_value_and_grad(self, cbf_policy: Policy, x0: State, include_h0: bool = False):
        ...

    @jax.jit
    def get_cbf_constr_next(self, cbf_policy: Policy, x0: State, u_nom, hH_dstb, alpha: float):
        def cbf_constr_at_x1(u0: Control, hh):
            V_at_x1, Vdot_at_x1 = self.Vdot_at_x1(u0, cbf_policy, x0, hH_dstb[hh], hh)
            return Vdot_at_x1 + alpha * V_at_x1

        # Since we are going to be calling a jac on the outside, do the same order as is done in hessian: jacfwd(jacrev)
        def get_constr_grads(hh: int):
            return jax.value_and_grad(cbf_constr_at_x1)(u_nom, hh)


        h_constr, hu_dconstr = jax.vmap(get_constr_grads)(jnp.arange(self.task.nh))

        assert h_constr.shape == (self.task.nh,)
        assert hu_dconstr.shape == (self.task.nh, self.task.nu)

        # Fold in the unom:
        #   c(u_nom) + ∇c(u_nom) (u - u_nom) <= 0
        #   [ c(u_nom) - ∇c(u_nom) u_nom ] + ∇c(u_nom) u <= 0
        h_constr_shift = h_constr - hu_dconstr @ u_nom
        logger.info("get_constr_next!")
        return h_constr_shift, hu_dconstr

    @jax.jit
    def get_Vdot_next(self, cbf_policy: Policy, x0: State, u_nom, hH_dstb):
        def Vdot_at_x1(u0: Control, hh):
            _, Vdot_at_x1_ = self.Vdot_at_x1(u0, cbf_policy, x0, hH_dstb[hh], hh)
            return Vdot_at_x1_

        def get_constr_grads(hh: int):
            return jax.value_and_grad(Vdot_at_x1)(u_nom, hh)

        # Since we are going to be calling a jac on the outside, do the same order as is done in hessian: jacfwd(jacrev)
        # h_Vdot, hu_dVdot = value_and_jacfwd(Vdot_at_x1)(u_nom)
        h_Vdot, hu_dVdot = jax.vmap(get_constr_grads)(jnp.arange(self.task.nh))
        assert h_Vdot.shape == (self.task.nh,)
        assert hu_dVdot.shape == (self.task.nh, self.task.nu)

        # Fold in the unom:
        #   c(u_nom) + ∇c(u_nom) (u - u_nom) <= 0
        #   [ c(u_nom) - ∇c(u_nom) u_nom ] + ∇c(u_nom) u <= 0
        h_Vdot_shift = h_Vdot - hu_dVdot @ u_nom
        return h_Vdot_shift, hu_dVdot

    def generate_piecewise_constant_disturbances(self, key, n_samples=50, horizon=10, nd=1, interval_size=2):

        n_full_intervals = horizon // interval_size
        if horizon % n_full_intervals != 0:
            n_intervals = n_full_intervals + 1
        else: n_intervals = n_full_intervals

        bi_pw_dstb = self.sample_dstb(key, n_samples, n_intervals, nd)

        expand_fn = ft.partial(self.expand_pw_dstb, horizon=horizon, interval_size=interval_size)
        bhH_pw_const_dstb = jax_vmap(expand_fn, rep=1)(bi_pw_dstb)

        return bhH_pw_const_dstb, bi_pw_dstb

    def expand_pw_dstb(self, i_pw_dstb, horizon, interval_size):
        H_pw_const_dstb = jnp.concatenate([jnp.repeat(i_pw_dstb[i:i+1, :], interval_size, axis=0) for i in range(i_pw_dstb.shape[0])],axis=0)

        H_pw_const_dstb=H_pw_const_dstb[ :horizon, :]

        return H_pw_const_dstb

    def Vdot_at_x1(self, control0: Control, cbf_policy: Policy, x0: State, H_dstb: Disturb, hh: int):
        """Same as rollout, but we apply control0 for one step, then the policy for the rest."""

        def body(carry, dstb):
            x = carry
            x_new = self.task.step_with_policy(x, cbf_policy, dstb)
            return x_new, x_new

        # h_hmax0 = self.task.h_vec(x0)
        x1 = self.task.step_with_control(x0, control0, H_dstb[0])

        def compute_value_from_x1(x1_):
            carry_init = x1_
            x_final, H_xnew = lax.scan(body, carry_init, H_dstb[1:])
            assert H_xnew.shape == (horizon - 1, self.task.nx)

            Hp1_xnew = jnp.concatenate([x1_[None], H_xnew], axis=0)
            assert Hp1_xnew.shape == (horizon, self.task.nx)

            Hh_hnew = jax.vmap(self.task.h_vec)(H_xnew)
            H_hnew = Hh_hnew[:, hh]
            hmax1 = self.task.h_vec(x1_)[hh]
            Hp1_h = jnp.concatenate([hmax1[None], H_hnew], axis=0)
            assert Hp1_h.shape == (horizon,)

            Hp1_t = jnp.linspace(0, 1, Hp1_h.shape[0])
            hmax_from1, _ = max_cubic_spl_fast(Hp1_t, Hp1_h, include_initial=True)

            return hmax_from1

        def tmp_fn(t_: float):
            """tmp fn whose gradient is Vx(x_1)^T ( f + G u_0 )"""
            dstb1 = H_dstb[1]
            x_tmp = x1 + (self.task.f(x1, dstb1) + self.task.G(x1, dstb1) @ control0) * t_
            return compute_value_from_x1(x_tmp)

        horizon = len(H_dstb)
        # Since we are going to be calling a jac on the outside, do the same order as is done in hessian: jacfwd(jacrev)
        # V_at_x1, Vdot_at_x1 = value_and_jacrev(tmp_fn)(0.0)
        V_at_x1, Vdot_at_x1 = jax.value_and_grad(tmp_fn)(0.0)
        return V_at_x1, Vdot_at_x1
