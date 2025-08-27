import pickle
import pathlib
from loguru import logger
from os import mkdir

from rpcbf.utils.data_compare_ci import CIData, CIDataBatchRollout,  CIDataGrid, CIDataGridRollout

def print_settings(data: CIData):

    print("Settings:")
    print("name: ", data.name)
    print("task_name: ", data.task_name)
    print("nom_pol: ", data.nom_pol)
    print("cbf_pol: ", data.cbf_pol)
    print("cbf_alpha: ", data.cbf_alpha)
    print("horizon: ", data.horizon)
    print("rollout_T: ", data.rollout_T)
    print("sampler: ", data.sampler)
    print("env_noise: ", data.env_noise)

def update_data(data_dir, data_name, ci_data_grid=None, ci_data_grid_rollout=None, ci_data_batch_rollout=None):

    pkl_path = pathlib.Path(data_dir / data_name)


    assert pkl_path.exists(), f"File {pkl_path} does not exist!"

    with open(pkl_path, "rb") as f:
        ci_data = pickle.load(f)

    logger.info("Loaded from {}!".format(pkl_path))

    if ci_data_grid is not None:
        ci_data.ci_data_grid = ci_data_grid
    if ci_data_grid_rollout is not None:
        ci_data.ci_data_grid_rollout = ci_data_grid_rollout
    if ci_data_batch_rollout is not None:
        ci_data.ci_data_batch_rollout = ci_data_batch_rollout

    with open(pkl_path, "wb") as f:
        pickle.dump(ci_data, f)


def save_data(data_dir, file_name, ci_data):
    # save data
    pkl_path = pathlib.Path(data_dir / file_name)
    if not data_dir.exists():
        mkdir(data_dir)

    with open(pkl_path, "wb") as f:
        pickle.dump(ci_data, f)

def load_data(data_dir, file_name):
    pkl_path = pathlib.Path(data_dir / file_name)
    with open(pkl_path, "rb") as f:
        ci_data = pickle.load(f)

    logger.info("Loaded from {}!".format(pkl_path))
    return ci_data


def set_data(task, file_name, nom_pol_str, cbf_pol_str, cbf_alpha, horizon, rollout_T, sampler_str, env_noise_str, ci_data=None):

    ci_data = CIData(name=file_name,
                     task_name=task.name,
                     nom_pol=nom_pol_str,
                     cbf_pol=cbf_pol_str,
                     cbf_alpha=cbf_alpha,
                     horizon=horizon,
                     rollout_T=rollout_T,
                     sampler=sampler_str,
                     env_noise=env_noise_str,
                     ci_data=ci_data)

    return ci_data
