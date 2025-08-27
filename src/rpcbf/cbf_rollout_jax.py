import diffrax
import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
from attrs import define
from jaxproxqp.jaxproxqp import JaxProxQP, QPSolution
from jaxtyping import PRNGKeyArray
from og.dyn_types import State
from og.jax_utils import concat_at_front, jax_vmap

from rpcbf.dyn.task import Policy, Task
from rpcbf.env_noise import UniformEnvNoise
from rpcbf.sampler.uniform_sampler import Sampler


class CBFRolloutJax:
    @define
    class Cfg:
        cbf_alpha: float

    def __init__(self, task: Task, env_noise: UniformEnvNoise, cfg: Cfg):
        self.task = task
        self.env_noise = env_noise
        self.cfg = cfg

        self.step_fn = jax.jit(task.step_with_control)

    def solve_cbf_qp(
        self, u_nom: jnp.ndarray, h_Vh: jnp.ndarray, grad_h_Vh: jnp.ndarray, h_f: jnp.ndarray, h_G: jnp.ndarray
    ):
        nu, nh, nx = self.task.nu, self.task.nh, self.task.nx

        # 0.5 ||u - u_nom||^2 = 0.5 u^T u - u^T u_nom + 0.5 u_nom^T u_nom = 0.5 u^T P u + q^T u + const
        H = jnp.eye(nu)
        g = jnp.array(-u_nom)

        # Vx^T (f + Gu) + alpha V <= 0
        # Vx^T G u <= -Vx^T f - alpha V
        def get_C_ub(Vx, f, G, Vh):
            C_single = ei.einsum(Vx, G, "nx, nx nu -> nu")
            ub_single = -(jnp.dot(Vx, f) + self.cfg.cbf_alpha * Vh)
            return C_single, ub_single

        C, ub = jax_vmap(get_C_ub)(grad_h_Vh, h_f, h_G, h_Vh)

        l_box = jnp.array(self.task.u_min)
        u_box = jnp.array(self.task.u_max)

        settings = JaxProxQP.Settings.default_float64()
        problem = JaxProxQP.QPModel.create(H, g, C, ub - 1e-3, l_box, u_box)
        solver = JaxProxQP(problem, settings)
        sol: QPSolution = solver.solve()
        u_qp = sol.x
        info = {}
        return u_qp, info

    def rollout(self, key_base: PRNGKeyArray, nom_pol: Policy, x0: State, rollout_T: int):
        def body(carry, kk):
            x_, pbar_state = carry
            # Compute value function.
            B, grad_B, h_f, h_G = self.task.hocbf_get_value_and_grad(x_)

            # Solve CBF-QP
            u_nom = nom_pol(x_)
            u_cbf, info_qp = self.solve_cbf_qp(u_nom, B, grad_B, h_f, h_G)

            # Step forward.
            dstb_env = self.env_noise.sample_dstb(key_base, x_, kk)
            x_new = self.step_fn(x_, u_cbf, dstb_env)

            # info = info_sampler | info_qp
            info = {}

            # [0, 1]
            progress_frac = (kk + 1) / rollout_T
            pbar_state = progress_meter.step(pbar_state, progress_frac)

            return (x_new, pbar_state), (x_new, u_nom, u_cbf, dstb_env, info)

        progress_meter = diffrax.TextProgressMeter()
        pbar_state0 = progress_meter.init()
        T_kk = jnp.arange(rollout_T)
        carry_init = (x0, pbar_state0)
        (_, pbar_state_final), (T_xnew, T_u_nom, T_u_cbf, T_dstb_env, T_info) = lax.scan(
            body, carry_init, T_kk, length=rollout_T
        )

        Tp1_x = concat_at_front(x0, T_xnew)
        Tp1h_h = jax_vmap(self.task.h_vec)(Tp1_x)
        progress_meter.close(pbar_state_final)

        return Tp1_x, T_u_cbf, T_u_nom, Tp1h_h, T_dstb_env, T_info
