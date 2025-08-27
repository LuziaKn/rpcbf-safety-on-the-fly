import jax
import numpy as np
import jax.numpy as jnp
from og.dyn_types import State
from jaxtyping import PRNGKeyArray

from rpcbf.dyn.task import Policy, Task
from rpcbf.env_noise import UniformEnvNoise


class RegRolloutJax:
    def __init__(self, task: Task, env_noise: UniformEnvNoise):
        self.task = task
        self.env_noise = env_noise
        self.step_fn = jax.jit(task.step_with_control)

    def rollout(self, key_base: PRNGKeyArray, nom_pol: Policy, x0: State, rollout_T: int):
        Tp1_x = [jnp.array(x0)]
        T_u_nom, T_u_cbf = [], []
        x = x0

        for kk in range(rollout_T):
            dstb_env = self.env_noise.sample_dstb(key_base, x, kk)
            # Step forward.
            u_nom = nom_pol(x)
            x = self.step_fn(x, u_nom, dstb_env )
            x = jnp.array(x)

            Tp1_x.append(x)
            T_u_nom.append(u_nom)

        Tp1_x = jnp.stack(Tp1_x, axis=0)
        T_u_nom = jnp.stack(T_u_nom, axis=0)

        return Tp1_x, T_u_nom
