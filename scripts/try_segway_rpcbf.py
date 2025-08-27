import functools as ft
import pickle

import ipdb

import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from flax import nnx
from loguru import logger
from og.jax_utils import jax_jit_np, jax_vmap
from og.plot_utils import line_labels
from og.tree_utils import tree_stack

from rpcbf.pcbf_rollout import PCBFRollout
from rpcbf.pcbf_rollout_jax import PCBFRolloutJax
from rpcbf.dyn.segway import Segway
from rpcbf.env_noise import PiecewiseBangBangEnvNoise, UniformEnvNoise, ZeroEnvNoise
from rpcbf.reg_rollout import RegRollout
from rpcbf.sampler.uniform_bang_bang_sampler import UniformBangBangSampler
from rpcbf.sampler.zero_sampler import ZeroSampler
from rpcbf.utils.path_utils import get_plot_dir

import jax
jax.config.update("jax_enable_x64", True)

def main():
    plot_dir = get_plot_dir() / "segway"

    task = Segway()

    horizon_s = 20  # horizon pcbf [s]
    rollout_T_s = 20  # horizon evaluation rollout [s]

    horizon = int(np.ceil(horizon_s / task.dt))
    rollout_T = int(np.ceil(rollout_T_s / task.dt))


    cbf_pol = task.nom_pol_lqr
    nom_pol = task.nom_pol_one

    rngs = nnx.Rngs(12345)

    n_samples = 100
    n_samples_plot = 50
    n_samples_uniform = 50
    piecewise_interval_size = 25

    logger.info("horizon: {}, rollout_T: {}, n_samples: {}".format(horizon, rollout_T, n_samples))

    sampler_cfg = UniformBangBangSampler.Cfg(n_samples, n_samples_uniform, piecewise_interval_size)
    sampler = UniformBangBangSampler.create(rngs(), task, horizon, sampler_cfg)

    sampler_plot_cfg = UniformBangBangSampler.Cfg(n_samples_plot, n_samples_uniform, piecewise_interval_size)
    sampler_plot = UniformBangBangSampler.create(rngs(), task, horizon, sampler_plot_cfg)

    b_x = np.linspace(-2.5, 2.5, num=64)
    b_y = np.linspace(-1.1, 1.1, num=64)
    bb_X, bb_Y = np.meshgrid(b_x, b_y)
    zero = np.zeros_like(bb_X)
    bb_state = np.stack([bb_X, bb_Y, zero, zero], axis=-1)

    def get_Vh(x0):
        _, h_hmax, _ = sampler_plot.get_value(cbf_pol, x0, include_h0=True)
        return h_hmax

    bbh_Vh = jax_jit_np(jax_vmap(get_Vh, rep=2))(bb_state)
    logger.info("Done getting contour")

    bb1_Vhmax = np.max(bbh_Vh, axis=-1)
    bbh_Vh_all = np.concatenate([bb1_Vhmax[..., None], bbh_Vh], axis=-1)

    # -----------------
    # Rollouts.
    logger.info("Generate rollouts")
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    rollouter = RegRollout(task)
    Tp1_x, T_u = rollouter.rollout(nom_pol, x0, rollout_T)

    env_noise = PiecewiseBangBangEnvNoise(task, piecewise_interval_size=5)

    cbf_cfg = PCBFRollout.Cfg(cbf_alpha=2.0)

    n_rollout_test = 25
    b_keys = jr.split(rngs(), n_rollout_test)

    if n_rollout_test > 1:
        rollouter = PCBFRolloutJax(task, sampler, env_noise, cbf_cfg)
        rollout_fn = ft.partial(rollouter.rollout, cbf_pol=cbf_pol, nom_pol=nom_pol, x0=x0, rollout_T=rollout_T)
        bTp1_x_cbf, bT_u_cbf, bT_u_nom, bTh_h, bT_dstb_env, bT_dstb, bT_info = jax_jit_np(jax_vmap(rollout_fn))(b_keys)
    else:
        rollouter = PCBFRollout(task, sampler, env_noise, cbf_cfg)
        b_out = []
        for ii in tqdm.trange(n_rollout_test):
            key_rollout = b_keys[ii]
            out = rollouter.rollout(key_rollout, cbf_pol, nom_pol, x0, rollout_T)
            b_out.append(out)

        b_out = tree_stack(b_out, axis=0)
        bTp1_x_cbf, bT_u_cbf, bT_u_nom, bTh_h, bT_dstb_env, bT_dstb, bT_info = b_out

    b_h = bTh_h.max(axis=(1, 2))

    # The sampler rollouts at each step.
    # bTHp1h_x_cbfpol = bT_info["Hp1_x"]
    plot_cbfpol_every = 50
    plot_cbfpol_idxs = np.arange(rollout_T)[::-1][::plot_cbfpol_every][::-1]

    # Plot
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

        cm = ax.contourf(bb_X, bb_Y, bb_Vh_masked, levels=8, cmap=cmap, alpha=0.7)
        zeroline = ax.contour(bb_X, bb_Y, bb_Vh, levels=[0], colors=["magenta"], zorder=3.5, linewidths=0.5, alpha=0.6)
        cbar = fig.colorbar(cm, ax=ax)
        cbar.add_lines(zeroline)


        ax.plot(Tp1_x[:, 0], Tp1_x[:, 1], zorder=5, color="C0")
        for Tp1_x_cbf, Vh_max_cbf in zip(bTp1_x_cbf, b_h):
            if Vh_max_cbf <= 0:
                c = "C4"
            else:
                c = "C3"
            ax.plot(Tp1_x_cbf[:, 0], Tp1_x_cbf[:, 1], zorder=5, color=c, lw=0.4)

        # plot initial point as circle
        ax.scatter(x0[0], x0[1], s=3**2, zorder=6, marker="o", color="C4")

        if ii == 0:
            ax.set_title("Total")
        else:
            ax.set_title(task.h_labels[ii - 1])

    fig_path = plot_dir / "rpcbf_contour.pdf"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------
    n_safe = len([1 for i in b_h if i <= 0])
    print(f"Safe: {n_safe}/{n_rollout_test}")
    print(f"Plot saved to {fig_path}")



if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
