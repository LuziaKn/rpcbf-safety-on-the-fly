import functools as ft
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from og.dyn_types import Control, Disturb, HFloat, State
from og.shape_utils import assert_shape

from rpcbf.dyn.odeint import rk4_multi


class Policy(Protocol):
    def __call__(self, state: State) -> Control: ...


class Task(Protocol):
    NX = None
    NU = None
    ND = None

    @property
    def name(self):
        return self.__class__.__name__

    def chk_x(self, state: State):
        return assert_shape(state, self.nx, "state")

    def chk_u(self, control: Control):
        return assert_shape(control, self.nu, "control")

    @property
    def nx(self) -> int:
        return self.NX

    @property
    def nu(self) -> int:
        return self.NU

    @property
    def nd(self) -> int:
        return self.ND

    @property
    def u_min(self):
        return np.full(self.nu, -1.0)

    @property
    def u_max(self):
        return np.full(self.nu, +1.0)

    def h_dict(self, state: State) -> dict[str, HFloat]: ...

    def h_vec(self, state: State) -> HFloat:
        h_dict = self.h_dict(state)
        assert len(h_dict) == len(self.h_labels)
        return jnp.array(list(h_dict.values()))

    @property
    def h_labels(self) -> list[str]: ...

    @property
    def nh(self) -> int:
        return len(self.h_labels)

    def f(self, state: State, dstb: Disturb) -> State: ...

    def G(self, state: State, dstb: Disturb): ...

    def xdot(self, state: State, control: Control, dstb: Disturb) -> State:
        self.chk_x(state)
        self.chk_u(control)
        control = control.clip(-1, 1)
        f, G = self.f(state, dstb), self.G(state, dstb)
        self.chk_x(f)
        Gu = G @ control
        self.chk_x(Gu)
        dx = f + Gu
        return self.chk_x(dx)

    @property
    def dt(self) -> float:
        """How long the disturbance is held constant for."""
        ...

    @property
    def n_steps(self) -> int:
        """How many steps to per dt when integrating."""
        ...

    def step_with_control(self, state: State, control: Control, dstb: Disturb) -> State:
        def constant_pol(x: State):
            return control

        return self.step_with_policy(state, constant_pol, dstb)

    def step_with_policy(self, state: State, pol: Policy, dstb: Disturb) -> State:
        """Piecewise constant disturbance"""

        def xdot_with_u(x: State):
            u = pol(x)
            return self.xdot(x, u, dstb)

        return rk4_multi(self.dt, xdot_with_u, state, self.n_steps)
