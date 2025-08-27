import functools as ft

import ipdb
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from flax import nnx
from loguru import logger
from matplotlib.colors import CenteredNorm
from og.jax_utils import jax_jit_np, jax_vmap

from rpcbf.pcbf_rollout import PCBFRollout
from rpcbf.pcbf_rollout_jax import PCBFRolloutJax
from rpcbf.dyn.doubleint_wall import DoubleIntWall
from rpcbf.env_noise import UniformEnvNoise
from rpcbf.reg_rollout import RegRollout
from rpcbf.sampler.uniform_bang_bang_sampler import UniformBangBangSampler
from rpcbf.sampler.uniform_sampler import UniformSampler
from rpcbf.sampler.zero_sampler import ZeroSampler
from rpcbf.utils.path_utils import get_plot_dir


def main():
    plot_dir = get_plot_dir() / "dbint"
    plot_dir.mkdir(exist_ok=True, parents=True)

    horizon = 64  # horizon pcbf
    rollout_T = 80  # horizon evaluation rollout
    n_samples = 100
    n_samples_uniform = 50
    piecewise_interval_size = 1

    logger.info("horizon: {}, rollout_T: {}, n_samples: {}".format(horizon, rollout_T, n_samples))

    rngs = nnx.Rngs(12345)

    task = DoubleIntWall()

    sampler_cfg = UniformBangBangSampler.Cfg(n_samples, n_samples_uniform, piecewise_interval_size)
    sampler = UniformBangBangSampler.create(rngs(), task, horizon, sampler_cfg)

    b_x = np.linspace(-1.5, 1.5, num=64)
    b_y = np.linspace(-3.0, 3.0, num=64)
    bb_X, bb_Y = np.meshgrid(b_x, b_y)
    bb_state = np.stack([bb_X, bb_Y], axis=-1)

    def get_Vh(x0):
        _, h_hmax, _ = sampler.get_value(task.nom_pol_osc, x0, include_h0=True)
        return h_hmax

    bbh_Vh = jax_jit_np(jax_vmap(get_Vh, rep=2))(bb_state)

    bb1_Vhmax = np.max(bbh_Vh, axis=-1)
    bbh_Vh_all = np.concatenate([bb1_Vhmax[..., None], bbh_Vh], axis=-1)

    # -----------------
    # Rollouts.
    cbf_pol = task.nom_pol_osc
    nom_pol = task.nom_pol_zero

    x0 = np.array([0, 0.9])


    rollouter = RegRollout(task)
    Tp1_x, T_u = rollouter.rollout(nom_pol, x0, rollout_T)

    env_noise = UniformEnvNoise(task)

    n_rollout_test = 25
    b_keys = jr.split(rngs(), n_rollout_test)
    cbf_cfg = PCBFRolloutJax.Cfg(cbf_alpha=1.0)
    rollouter = PCBFRolloutJax(task, sampler, env_noise, cbf_cfg)
    rollout_fn = ft.partial(rollouter.rollout, cbf_pol=cbf_pol, nom_pol=nom_pol, x0=x0, rollout_T=rollout_T)

    bTp1_x_cbf, _, _, bT_Vh_max_cbf, bT_dstb_env, bThH_dstb, bH_info = jax_jit_np(jax_vmap(rollout_fn))(b_keys)
    b_Vh_max_cbf = bT_Vh_max_cbf.max(axis=(1, 2))

    # check for nans
    nans_found = np.any(np.isnan(bTp1_x_cbf))

    # -----------------
    # Plot

    figure = plt.figure()
    test_id = 2
    ax = figure.add_subplot(111)
    ax.plot(bThH_dstb[test_id, :,0,0, 0], label="Disturbance")
    ax.plot(bT_dstb_env[test_id, :, 0], label="Env Disturbance")
    ax.legend()
    fig_path = plot_dir / "dstb.pdf"
    figure.savefig(fig_path, bbox_inches="tight")
    plt.close(figure)

    cmap = "crest"

    ncol = task.nh + 1
    figsize = np.array([ncol * 4, 4])
    axes: list[plt.Axes]
    fig, axes = plt.subplots(1, ncol, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        task.setup_plot(ax=ax)

        bb_Vh = bbh_Vh_all[:, :, ii]
        # Only plot the contours of the safe region.
        bb_Vh_masked = np.ma.array(bb_Vh, mask=bb_Vh > 0.0)

        cm = ax.contourf(bb_X, bb_Y, bb_Vh_masked, levels=8, cmap=cmap)
        zeroline = ax.contour(bb_X, bb_Y, bb_Vh, levels=[0], colors=["magenta"], zorder=3.5)
        cbar = fig.colorbar(cm, ax=ax)
        cbar.add_lines(zeroline)

        ax.plot(Tp1_x[:, 0], Tp1_x[:, 1], zorder=5, color="C0")
        for (Tp1_x_cbf, Vh_max_cbf) in zip(bTp1_x_cbf, b_Vh_max_cbf):
            if Vh_max_cbf <= 0:
                c = "C4"
            else:
                c = "C3"
            ax.plot(Tp1_x_cbf[:, 0], Tp1_x_cbf[:, 1], zorder=5, color=c, lw=0.6)

        # plot initial point as circle
        ax.scatter(x0[0], x0[1], s=3**2, zorder=6, marker="o", color="C4")

        ax.set_xlabel("position")
        ax.set_ylabel("velocity")

        if ii == 0:
            ax.set_title("Total")
        else:
            ax.set_title(task.h_labels[ii - 1])

    fig_path = plot_dir / "rpcbf_contour.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print("Final x: {}".format(Tp1_x_cbf[-1]))

    if nans_found:
        print(f"Found nans in trajectories")

    n_safe = len([1 for i in b_Vh_max_cbf if i <= 0])
    print(f"Safe: {n_safe}/{n_rollout_test}")
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
