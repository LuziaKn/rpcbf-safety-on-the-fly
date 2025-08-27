import jax
import numpy as np
from og.dyn_types import State

from rpcbf.dyn.task import Policy, Task


class RegRollout:
    def __init__(self, task: Task):
        self.task = task
        self.step_fn = jax.jit(task.step_with_control)

    def rollout(self, nom_pol: Policy, x0: State, rollout_T: int):
        Tp1_x = [np.array(x0)]
        T_u_nom, T_u_cbf = [], []
        x = x0

        dstb = np.zeros((self.task.nd))

        for kk in range(rollout_T):
            # Step forward.
            u_nom = nom_pol(x)
            x = self.step_fn(x, u_nom, dstb)
            x = np.array(x)

            Tp1_x.append(x)
            T_u_nom.append(u_nom)

        Tp1_x = np.stack(Tp1_x, axis=0)
        T_u_nom = np.stack(T_u_nom, axis=0)



        return Tp1_x, T_u_nom
