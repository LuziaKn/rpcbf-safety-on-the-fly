from typing import NamedTuple
import einops as ei

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import shapely
from og.dyn_types import Disturb, HFloat, State
from og.plot_phase_utils import plot_x_bounds, plot_y_bounds
from og.remap import remap_box
from og.small_la import inv22

from rpcbf.dyn.task import Task
from rpcbf.policies import ConstantPolicy, LinearPolicy
from rpcbf.utils.hocbf import hocbf


class Segway(Task):
    """Disturbance acts as a mass."""

    NX = 4
    NU = 1
    ND = 1

    hocbf_alpha = 4.0

    P, TH, V, W = range(NX)
    (F,) = range(NU)

    #DT_ORIG = 0.02
    DT_ORIG = 0.01
    # DT_ORIG = 0.008
    # DT_ORIG = 0.008
    # DT_ORIG = 0.006
    # DT_ORIG = 0.005
    # DT_ORIG = 0.002
    # DT_ORIG = 0.001
    DT_MULT = 10.0

    class Params(NamedTuple):
        m: float = 0.5
        M: float = 1.5
        J: float = 0.01
        l: float = 1.0
        g: float = 9.81
        # Linear friction
        c: float = 0.1
        # Angular friction
        gamma: float = 0.1

    def __init__(self, p=Params(), dt: float = None):
        self.p = p
        self.umax = 40.0
        self._theta_max = 0.3 * np.pi  # 0.94
        self._p_max = 2.0

        self.dt_mult = Segway.DT_MULT

        self._dt_orig = Segway.DT_ORIG if dt is None else dt
        self._dt = self._dt_orig * self.dt_mult
        self._p_max = 2.0

        self.m_lo = 0.1
        self.m_hi = 0.9

        with jax.ensure_compile_time_eval():
            # x_eq, u_eq = np.zeros(self.nx), np.zeros(self.nu)
            # dstb_zero = np.zeros(self.nd)
            # # Q = np.diag([0.1, 1.0, 5.0, 2.0])
            # Q = np.diag([1.0, 1.0, 5.0, 1.0])
            # R = 20 * np.diag(np.ones(self.nu))
            # logger.info("Getting jac of f")
            # A, B = jax2np(jax.jacobian(self.f)(x_eq, dstb_zero)), jax2np(self.G(x_eq, dstb_zero))
            # logger.info("Solving continuous are...")
            # S = scipy.linalg.solve_continuous_are(A, B, Q, R)
            # logger.info("Solving continuous are... Done!")
            # self.lqr_K = np.linalg.inv(R) @ B.T @ S
            # print()
            # print("lqr_K:")
            # print(repr(self.lqr_K))
            self.lqr_K = np.array([[-0.2236068, 4.81717023, -0.67126635, 1.48039908]])
            # print()
        # exit(0)

    @property
    def dt(self):
        return self._dt

    @property
    def n_steps(self) -> int:
        return 1

    def M_mat(self, state: State, p: Params):
        pos, th, v, w = self.chk_x(state)
        M_11 = p.m + p.M
        M_12 = M_21 = -p.m * p.l * jnp.cos(th)
        M_22 = p.J + p.m * p.l**2
        return jnp.array([[M_11, M_12], [M_21, M_22]])

    def dstb_to_mass(self, dstb: Disturb):
        (m_dstb,) = dstb
        # Map [-1, 1] to [invm_lo, invm_hi]
        mass = remap_box(m_dstb, self.m_lo, self.m_hi)
        return mass

    def f(self, state: State, dstb: Disturb) -> State:
        p = self.p

        m_new = self.dstb_to_mass(dstb)
        p = p._replace(m=m_new)

        _, th, v, w = self.chk_x(state)
        sin, cos = jnp.sin(th), jnp.cos(th)

        Ctau = jnp.array([p.c * v + p.m * p.l * sin * w**2, p.gamma * w - p.m * p.g * p.l * sin])
        M_inv = inv22(self.M_mat(state, p))
        F_vel = -M_inv @ Ctau
        # F_vel = jnp.linalg.solve(self.M_mat(state), -Ctau)
        assert F_vel.shape == (2,)
        F = jnp.concatenate([jnp.array([v, w]), F_vel], axis=0)
        assert F.shape == (self.nx,)
        return F / self.dt_mult

    def G(self, state: State, dstb: Disturb):
        p = self.p
        m_new = self.dstb_to_mass(dstb)
        p = p._replace(m=m_new)

        self.chk_x(state)
        B = np.array([[1.0, 0.0]]).T
        M_inv = inv22(self.M_mat(state, p))
        # G_vel = jnp.linalg.solve(self.M_mat(state), B)
        G_vel = M_inv @ B
        assert G_vel.shape == (2, self.nu)
        G = jnp.concatenate([jnp.zeros((2, self.nu)), G_vel], axis=0)
        assert G.shape == (self.nx, self.nu)
        return G * self.umax / self.dt_mult

    def h_dict(self, state: State) -> dict[str, HFloat]:
        self.chk_x(state)
        p, th, v, w = state

        h_theta_ub = -(th + self._theta_max)
        h_theta_lb = th - self._theta_max

        h_p_ub = p - self._p_max
        h_p_lb = -(p + self._p_max)

        return {"theta_ub": h_theta_ub, "theta_lb": h_theta_lb, "p_ub": h_p_ub, "p_lb": h_p_lb}

    @property
    def state_lims(self):
        return {Segway.P: [-self._p_max, self._p_max], Segway.TH: [-self._theta_max, self._theta_max]}

    @property
    def h_labels(self) -> list[str]:
        return ["theta_ub", "theta_lb", "ub", "lb"]

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$\theta$", r"$v$", r"$\omega$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$f$"]

    @property
    def nom_pol_lqr(self):
        # lqr_K = np.array([[-1.41421356, 14.7225004, -2.47300348, 4.76056567]])
        # lqr_K = np.array([[-0.70710678,  4.80752431, -0.82246958,  1.46210067]])
        # lqr_K = np.array([[-0.70710678,  9.88421616, -1.53684284,  3.2050113 ]])
        lqr_K = self.lqr_K
        return LinearPolicy.create(lqr_K)

    @property
    def nom_pol_zero(self):
        u = np.zeros(self.nu)
        return ConstantPolicy.create(u)

    @property
    def nom_pol_one(self):
        u = np.ones(self.nu)
        return ConstantPolicy.create(u)

    def setup_plot(self, ax: plt.Axes):
        return self.setup_plot_ptheta(ax)

    def setup_plot_ptheta(self, ax: plt.Axes):
        """x-axis is position, y-axis is theta."""
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.1, 1.1)

        # Plot the unsafe regions.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self._p_max, self._p_max), obs_style)
        plot_y_bounds(ax, (-self._theta_max, self._theta_max), obs_style)

        return ax

    def setup_plot_pv(self, ax: plt.Axes):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-3.0, 3.0)

        # Plot the unsafe regions.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self._p_max, self._p_max), obs_style)

    def setup_plot_pend(self, ax: plt.Axes):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-3.0, 3.0)

        # Plot the unsafe regions.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self._p_max, self._p_max), obs_style)

    def generate_grid(self, n_pts: int = 64):
        b_x = np.linspace(-2.2, 2.2, n_pts)
        b_y = np.linspace(-1.1, 1.1, n_pts)
        bb_X, bb_Y = np.meshgrid(b_x, b_y)
        zero = np.zeros_like(bb_X)
        bb_state = np.stack([bb_X, bb_Y, zero, zero], axis=-1)

        return bb_X, bb_Y, bb_state

    def plot_phase_paper(self, ax: plt.Axes):
        self.setup_plot_ptheta(ax)
        ax.set_xlabel(self.x_labels[0])
        ax.set_ylabel(self.x_labels[1], labelpad=-3.0)
        return ax

    def x0(self, idx: int) -> State:
        x0 = np.array([[-1, -0.5, 0.0, 0.0],
                       [0.0, 0.0, 0.0, -2.5]])
        return x0[idx]

    # def compute_hocbf_B(self, state: State) -> HFloat:
    #     p, th, v, w = self.chk_x(state)
    #     h_theta_ub, h_theta_lb, h_p_ub, h_p_lb = jnp.array(list(self.h_vec(state)))
    #
    #     theta_ub = w + self.hocbf_alpha * h_theta_ub
    #     theta_lb = - w + self.hocbf_alpha * h_theta_lb
    #     p_ub = v + self.hocbf_alpha * h_p_ub
    #     p_lb = -v + self.hocbf_alpha * h_p_lb
    #
    #     h_h = jnp.array([theta_lb, theta_ub, p_lb, p_ub])
    #     return h_h

    def handcbf_B(self, state: State) -> HFloat:
        p, th, v, w = self.chk_x(state)
        theta_lb = th - self._theta_max
        theta_ub = -(th + self._theta_max)
        p_lb = p - self._p_max
        p_ub = -(p + self._p_max)

        h_h = jnp.array([theta_lb, theta_ub, p_lb, p_ub])
        return h_h

    def compute_hocbf_B(self, state: State):
        self.chk_x(state)
        alpha0s = np.array([0.05, 0.05, 1.0, 1.0])
        dstb = np.zeros(self.nd)
        return hocbf(self.handcbf_B, self.f, alpha0s, state, dstb)


    def hocbf_get_value(self, x0: State):
        return self.compute_hocbf_B(x0)

    def hocbf_get_value_and_grad(self, x0: State):

        def compute_value(x0_):
            return self.compute_hocbf_B(x0_)

        B = self.compute_hocbf_B(x0)
        assert B.shape == (4,)


        grad_B = jax.jacobian(compute_value)(x0)
        assert grad_B.shape == (4,4)

        dstb_zero = np.zeros(self.nd)

        h_f = ei.repeat(self.f(x0, dstb_zero), "nx -> nh nx", nh=self.nh)
        h_G = ei.repeat(self.G(x0, dstb_zero), "nx nu -> nh nx nu", nh=self.nh)

        return B, grad_B, h_f, h_G
