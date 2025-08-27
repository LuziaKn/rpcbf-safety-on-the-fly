import einops as ei
import jax
import jax.random as jr
import numpy as np
import qpsolvers
import tqdm
from attrs import define
from jaxtyping import PRNGKeyArray
from og.dyn_types import State
from og.jax_utils import jax_jit_np, jax_vmap
from og.tree_utils import tree_stack

from rpcbf.dyn.task import Policy, Task
from rpcbf.env_noise import UniformEnvNoise
from rpcbf.sampler.uniform_sampler import Sampler


class CBFRollout:
    @define
    class Cfg:
        cbf_alpha: float

    def __init__(self, task: Task, sampler: Sampler, env_noise: UniformEnvNoise, cfg: Cfg):
        self.task = task
        self.sampler = sampler
        self.env_noise = env_noise
        self.cfg = cfg

        self.step_fn = jax.jit(task.step_with_control)
        self.sample_dstb_fn = jax.jit(self.env_noise.sample_dstb)

    def solve_cbf_qp(self, u_nom: np.ndarray, h_Vh: np.ndarray, grad_h_Vh: np.ndarray, h_f, h_G):
        nu, nh, nx = self.task.nu, self.task.nh, self.task.nx

        # 0.5 ||u - u_nom||^2 = 0.5 u^T u - u^T u_nom + 0.5 u_nom^T u_nom = 0.5 u^T P u + q^T u + const
        P = np.eye(nu)
        q = -u_nom

        # Vx^T (f + Gu) + alpha V <= 0
        # Vx^T G u <= -Vx^T f - alpha V
        G_cbf = np.zeros((nh, nu))
        h_cbf = np.zeros(nh)

        for ii in range(nh):
            Vx = grad_h_Vh[ii]
            G_cbf[ii, :] = ei.einsum(Vx, h_G[ii], "nx, nx nu -> nu")
            h_cbf[ii] = -(np.dot(Vx, h_f[ii]) + self.cfg.cbf_alpha * h_Vh[ii])

        G_qp, h_qp = G_cbf, h_cbf

        lb = self.task.u_min
        ub = self.task.u_max

        problem = qpsolvers.Problem(P, q, G_qp, h_cbf, lb=lb, ub=ub)
        solution = qpsolvers.solve_problem(problem, solver="proxqp")
        u_qp = solution.x
        return u_qp

    def rollout(self, key_base: PRNGKeyArray, cbf_pol: Policy, nom_pol: Policy, x0: State, rollout_T: int):
        Tp1_x = [np.array(x0)]
        T_u_nom, T_u_cbf = [], []
        T_dstb_env, HhT_dstb = [], []
        H_info = []
        x = x0

        for kk in tqdm.trange(rollout_T):
            # Compute value function.
            self.sampler, hT_dstb, h_Vh, grad_h_Vh, h_f, h_G, info = self.sampler.get_value_and_grad(cbf_pol, x)
            h_Vh, grad_h_Vh = np.array(h_Vh), np.array(grad_h_Vh)
            h_f, h_G = np.array(h_f), np.array(h_G)

            # Solve CBF-QP.
            u_nom = nom_pol(x)
            u_cbf = self.solve_cbf_qp(u_nom, h_Vh, grad_h_Vh, h_f, h_G)

            # Step forward.
            dstb_env = self.sample_dstb_fn(key_base, x, kk)
            dstb_env = np.array(dstb_env)
            x = self.step_fn(x, u_cbf, dstb_env)
            x = np.array(x)

            Tp1_x.append(x)
            T_u_nom.append(u_nom)
            T_u_cbf.append(u_cbf)
            T_dstb_env.append(dstb_env)
            HhT_dstb.append(hT_dstb)
            H_info.append(info)

        Tp1_x = np.stack(Tp1_x, axis=0)
        T_u_nom = np.stack(T_u_nom, axis=0)
        T_u_cbf = np.stack(T_u_cbf, axis=0)
        T_dstb_env = np.stack(T_dstb_env, axis=0)
        HhT_dstb = np.stack(HhT_dstb, axis=0)
        Tp1h_h = jax_jit_np(jax_vmap(self.task.h_vec, rep=1))(Tp1_x)
        H_info = tree_stack(H_info, axis=0)

        return Tp1_x, T_u_cbf, T_u_nom, Tp1h_h, T_dstb_env, HhT_dstb, H_info
