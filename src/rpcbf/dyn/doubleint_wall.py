import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import shapely
from og.dyn_types import Disturb, HFloat, State
from og.plot_phase_utils import plot_x_bounds
from og.jax_utils import jax_vmap
# from og.remap import remap_box

from rpcbf.dyn.task import Task
from rpcbf.policies import ConstantPolicy, LinearPolicy, SignedConstantPolicy
from rpcbf.plotting.plot_levelsets import poly_to_patch

def remap_box(x, lo, hi):
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x

class DoubleIntWall(Task):
    """Disturbance acts as a mass."""

    NX = 2
    NU = 1
    ND = 1

    P, V = range(NX)
    (A,) = range(NU)

    hocbf_alpha = 4.0

    def __init__(self):
        self.m_lo = 0.5
        self.m_hi = 2.0
        self.pos_wall = 1.0
        self._dt = 0.1


    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        # Correct: set the internal variable _dt
        self._dt = value


    @property
    def n_steps(self) -> int:
        return 1

    def dstb_to_invmass(self, dstb: Disturb):
        invm_lo = 1 / self.m_lo
        invm_hi = 1 / self.m_hi
        # Map [-1, 1] to [invm_lo, invm_hi]
        invmass = remap_box(dstb, invm_lo, invm_hi)
        return invmass

    def f(self, state: State, dstb: Disturb) -> State:
        self.chk_x(state)
        p, v = state
        #v_dstb, = dstb
        #v_dstb = 2 * v_dstb * self.dt
        v_dstb = v #+ v_dstb
        return jnp.array([v_dstb, 0.0])

    def G(self, state: State, dstb: Disturb):
        self.chk_x(state)
        m_dstb, = dstb
        GT = jnp.array([[0.0, 1.0 + 0.5*m_dstb]])  #[0, 1/m]
        G = GT.T
        return G

    def h_dict(self, state: State) -> dict[str, HFloat]:
        self.chk_x(state)
        p, v = state
        h_p_ub = p - 1.0
        h_p_lb = -(p + 1.0)
        return {"ub": h_p_ub, "lb": h_p_lb}


    @property
    def h_labels(self) -> list[str]:
        return ["ub", "lb"]

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$v$"]

    @property
    def nom_pol_zero(self):
        u = np.zeros(self.nu)
        return ConstantPolicy.create(u)

    @property
    def nom_pol_acc(self):
        u = np.array([0.5])
        return SignedConstantPolicy.create(u=u, id=1)

    @property
    def nom_pol_osc(self):
        K = np.array([[1.01, 0.2]])
        return LinearPolicy.create(K)

    @property
    def random_pol(self):
        key = jr.PRNGKey(123)
        u = jr.normal(key, (self.nu, self.nx))
        return u


    def setup_plot(self, ax: plt.Axes):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-3.0, 3.0)

        # Plot the unsafe regions.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)

    def compute_hocbf_B(self, x0: State) -> HFloat:
        p, v = x0
        h_p_ub, h_p_lb = jnp.array(list(self.h_vec(x0)))

        B_p_ub = v + self.hocbf_alpha * h_p_ub
        B_p_lb = -v + self.hocbf_alpha* h_p_lb

        B = jnp.stack([B_p_ub, B_p_lb], axis=0)

        return B

    def hocbf_get_value(self, x0: State):
        return self.compute_hocbf_B(x0)

    def hocbf_get_value_and_grad(self, x0: State):

        def compute_value(x0_):
            return self.compute_hocbf_B(x0_)

        B = self.compute_hocbf_B(x0)
        assert B.shape == (2,)


        grad_B = jax.jacobian(compute_value)(x0)
        assert grad_B.shape == (2,2)

        dstb_zero = np.zeros(self.nd)

        h_f = ei.repeat(self.f(x0, dstb_zero), "nx -> nh nx", nh=self.nh)
        h_G = ei.repeat(self.G(x0, dstb_zero), "nx nu -> nh nx nu", nh=self.nh)

        return B, grad_B, h_f, h_G

    def get_true_ci_points(self):
        # v**2 = v0**2 + 2 * a *delta_p
        # 0 = v0**2 + 2 * delta_p (a=1)
        # delta_p = -v0**2/2

        all_xs, all_vs = [], []
        vs = np.linspace(-2.0, 2.0)

        # pos v and pos a
        xs = self.pos_wall - np.maximum(0.0, vs)**2 / 2
        all_xs += [xs]
        all_vs += [vs]

        # neg v and max neg a
        vs = np.linspace(-2.0, 2.0)[::-1]
        xs = -self.pos_wall + np.minimum(0.0, vs)**2 / 2
        all_xs += [xs]
        all_vs += [vs]

        all_xs, all_vs = np.concatenate(all_xs), np.concatenate(all_vs)
        assert all_xs.ndim == all_vs.ndim == 1



        return np.stack([all_xs, all_vs], axis=1)

    def plot_phase_paper(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -1.25, 1.25
        PLOT_YMIN, PLOT_YMAX = -2.25, 2.25
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set_xlabel(self.x_labels[0])
        ax.set_ylabel(self.x_labels[1], labelpad=-3.0)

        outside_pts = [(PLOT_XMIN, PLOT_YMIN), (PLOT_XMIN, PLOT_YMAX), (PLOT_XMAX, PLOT_YMAX), (PLOT_XMAX, PLOT_YMIN)]
        outside = shapely.Polygon(outside_pts)

        # Plot the outside of the CI as a shaded region.
        ci_pts = self.get_true_ci_points()
        hole = shapely.Polygon(ci_pts)

        ci_poly = outside.difference(hole)
        patch = poly_to_patch(ci_poly, facecolor="0.6", edgecolor="none", alpha=0.5, zorder=3)
        ax.add_patch(patch)
        hatch_color = "0.5"
        # patch = poly_to_patch(
        #     ci_poly, facecolor="none", edgecolor=hatch_color, linewidth=0, zorder=3.1, hatch="."
        # )
        # ax.add_patch(patch)

        # Plot the obstacle.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)
        obs_style = dict(facecolor="none", lw=1.0, edgecolor="0.4", alpha=0.8, zorder=3.4, hatch="/")
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)

        return ax

    def generate_grid(self, n_pts=64):
        p = np.linspace(-1.1, 1.1, n_pts)
        v = np.linspace(-2.2, 2.2, n_pts)
        bb_P, bb_V = np.meshgrid(p, v)
        bb_state = np.stack([bb_P, bb_V], axis=-1)
        return bb_P, bb_V, bb_state

    def x0(self, idx: int):
        x0 = np.array([[0.0, 0.9],
                       [0.0, 1.35]])
        return x0[idx]
