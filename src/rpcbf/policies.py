import numpy as np
import jax.numpy as jnp
from flax import struct
from jaxtyping import ArrayLike, Float
from og.dyn_types import Control, State


class ConstantPolicy(struct.PyTreeNode):
    u: Control = struct.field(pytree_node=False)

    @staticmethod
    def create(u: Control):
        return ConstantPolicy(u)

    def __call__(self, x: State):
        return self.u

class SignedConstantPolicy(struct.PyTreeNode):
    u: Control = struct.field(pytree_node=False)
    id: int = struct.field(pytree_node=False)

    @staticmethod
    def create(u: Control, id: int):
        return SignedConstantPolicy(u, id)

    def __call__(self, x: State):
        return jnp.sign(x[self.id]) * self.u

class LinearPolicy(struct.PyTreeNode):
    K: Control = struct.field(pytree_node=False)
    x0: State | None = struct.field(pytree_node=False)
    u0: Control | None = struct.field(pytree_node=False)

    @staticmethod
    def create(K: Float[ArrayLike, "nu nx"], x0: State | None = None, u0: Control | None = None):
        return LinearPolicy(K, x0, u0)

    def __call__(self, x: State):
        dx = x
        if self.x0 is not None:
            dx = x - self.x0

        u = -self.K @ dx

        if self.u0 is not None:
            u = u + self.u0

        return u.clip(-1, 1)
