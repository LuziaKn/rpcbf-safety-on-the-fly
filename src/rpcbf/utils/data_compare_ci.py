from typing import Any, NamedTuple, Union

from og.dyn_types import BBTControl, BBTHFloat, BBTState, BBHFloat, BTState, BTControl
from og.jax_types import BBFloat
from og.jax_types import Arr, Float

BTDisturb = Float[Arr, "b1 T nd"]
BBTDisturb = Float[Arr, "b1 b2 T nd"]
BTHFloat = Float[Arr, "b T nh"]
BFloat = Float[Arr, "b1"]

class CIDataGrid(NamedTuple):
    bb_Xs: BBFloat
    bb_Ys: BBFloat

    bbh_Vh_all: BBHFloat

    horizon: int

    notes: dict[str, Any] = {}

class CIDataBatchRollout(NamedTuple):
    bb_Xs: BBFloat
    bb_Ys: BBFloat

    bTp1_x: BTState
    bT_u_cbf: BTControl

    bTp1_x_nom: BTState
    bT_u_nom: BTControl

    bTp1h_h: BTHFloat
    b_Vh: BFloat
    bT_dstb_env: BTDisturb
    bThH_dstb: BTDisturb #not corret

    horizon: int

    notes: dict[str, Any] = {}

class CIDataGridRollout(NamedTuple):
    bb_Xs: BBFloat
    bb_Ys: BBFloat

    bbTp1_x: BBTState
    bbTp1h_h: BBTHFloat
    bbT_u_cbf: BBTControl

    bbT_dstb_env: BBTDisturb
    bbT_dstb: BBTDisturb

    bbTp1_x_nom: BBTState
    bbT_u_nom: BBTControl

    horizon: int

    notes: dict[str, Any] = {}

class CIData(NamedTuple):
    name: str
    task_name: str
    nom_pol: str
    cbf_pol: str
    cbf_alpha: float
    horizon: int
    rollout_T: int
    sampler: str
    env_noise: str


    ci_data: Union[None, CIDataGrid, CIDataBatchRollout, CIDataGridRollout]

