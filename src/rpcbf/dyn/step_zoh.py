import jax
import numpy as np
from diffrax import Bosh3, ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve

from og.dyn_types import Control, State
from rpcbf.dyn.task import Task


def get_step_zoh(task: Task, dt: float, solver: str = "tsit5"):
    def step_zoh_(state: State, control: Control, dstb) -> Control:
        def body(t, state_, args):
            return task.xdot(state_, control, dstb)

        solver_name = solver
        solver_ = {"tsit5": Tsit5, "bosh3": Bosh3}[solver_name]()

        term = ODETerm(body)
        saveat = SaveAt(t1=True)
        stepsize_controller = ConstantStepSize()
        solution = diffeqsolve(
            term,
            solver_,
            t0=0,
            t1=dt,
            dt0=dt / 2,
            y0=state,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4,
        )

        assert len(solution.ys) == 1
        return solution.ys[0]

    jit_step_zoh_ = jax.jit(step_zoh_)

    def step_zoh(state: State, control: Control, dstb) -> Control:
        return np.array(jit_step_zoh_(state, control, dstb))

    return step_zoh