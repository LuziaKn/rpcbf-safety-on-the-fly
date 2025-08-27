import equinox as eqx
import jax
import jax.debug as jd
import jax.lax as lax
import numpy as np
from interpax import CubicSpline
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

DBG = False
TERMINATE = False


def max_cubic_spl_fast(
    b_x: Float[ArrayLike, "b"],
    b_y: Float[ArrayLike, "b"],
    bc_type: str | tuple = "not-a-knot",
    extrapolate: bool = False,
    check: bool = False,
    include_initial: bool = False,
    n_prev: int = 5,
):
    if DBG:
        b_y = eqx.debug.backward_nan(b_y, name="[max_cubic_spl_fast] b_y", terminate=TERMINATE)

    b = len(b_x)
    n_total = 2 * n_prev + 1
    assert b >= n_total
    # Start with the argmax.
    idx = jnp.argmax(b_y)
    s_idx = idx - n_prev
    e_idx = idx + n_prev + 1

    # Adjust if the s_idx or e_idx are out of bounds.
    #    If s_idx is negative, then add so s_idx = 0.
    idx_delta = jnp.maximum(0, -s_idx)
    #    If e_idx > b, then subtract so that e_idx = b.
    idx_delta -= jnp.maximum(0, e_idx - b)
    s_idx += idx_delta
    e_idx += idx_delta

    s_idx = eqx.error_if(s_idx, s_idx < 0, "s_idx < 0")
    s_idx = eqx.error_if(s_idx, e_idx > b, "e_idx > b")

    b_x_small = lax.dynamic_slice(b_x, (s_idx,), (n_total,))
    b_y_small = lax.dynamic_slice(b_y, (s_idx,), (n_total,))
    return max_cubic_spl(b_x_small, b_y_small, bc_type, extrapolate, check, include_initial)


def max_cubic_spl(
    b_x: Float[ArrayLike, "b"],
    b_y: Float[ArrayLike, "b"],
    bc_type: str | tuple = "not-a-knot",
    extrapolate: bool = False,
    check: bool = False,
    include_initial: bool = False,
):
    spl = CubicSpline(b_x, b_y, bc_type=bc_type, extrapolate=extrapolate, check=check)
    # if DBG:
    #     spl = eqx.debug.backward_nan(spl, name="CubicSpline", terminate=TERMINATE)

    def max_cubic(ii):
        x_l = spl.x[ii]
        x_r = spl.x[ii + 1]
        c = spl.c[:, ii]
        cder = jnp.polyder(c, m=1)
        if DBG:
            cder = eqx.debug.backward_nan(cder, "cder", terminate=TERMINATE)
        # Get the roots of the derivative.
        r1, r2, is_valid = quadratic_roots(cder)
        # if DBG:
        #     r1 = eqx.debug.backward_nan(r1, "r1", terminate=TERMINATE)
        #     r2 = eqx.debug.backward_nan(r2, "r2", terminate=TERMINATE)

        # We want a maximum, so check the second derivative to see if it is negative.
        cder2 = jnp.polyder(c, m=2)
        # if DBG:
        #     cder2 = eqx.debug.backward_nan(cder2, "cder2", terminate=TERMINATE)

        r1_neg = jnp.polyval(cder2, r1) < 0
        r2_neg = jnp.polyval(cder2, r2) < 0
        r1_between = (0 <= r1) & (r1 <= x_r - x_l)
        r2_between = (0 <= r2) & (r2 <= x_r - x_l)
        r1_good = r1_neg & r1_between & is_valid
        r2_good = r2_neg & r2_between & is_valid

        r1_maxval = jnp.where(r1_good, jnp.polyval(c, r1), -np.inf)
        r2_maxval = jnp.where(r2_good, jnp.polyval(c, r2), -np.inf)

        max_vals = jnp.array([r1_maxval, r2_maxval, jnp.array(b_y)[ii + 1]])
        max_locs = jnp.array([x_l + r1, x_l + r2, x_r])
        argmax_ = jnp.argmax(max_vals)
        return max_vals[argmax_], max_locs[argmax_]

    n_pts = len(b_x)
    b_maxval, b_maxloc = jax.vmap(max_cubic)(jnp.arange(n_pts - 1))
    argmax = jnp.argmax(b_maxval)
    maxval, maxloc = b_maxval[argmax], b_maxloc[argmax]

    if include_initial:
        # Finally, compare against the initial point.
        initial_val = b_y[0]
        initial_max = initial_val >= maxval
        maxval = jnp.where(initial_max, initial_val, maxval)
        maxloc = jnp.where(initial_max, b_x[0], maxloc)

    return maxval, maxloc


def quadratic_roots(coefs):
    """Solves for the roots of a quadratic polynomial in a numerically stable way."""
    assert coefs.shape == (3,)
    a, b, c = coefs

    if DBG:
        a = eqx.debug.backward_nan(a, "[quadratic roots] a", terminate=TERMINATE)
        b = eqx.debug.backward_nan(b, "[quadratic roots] b", terminate=TERMINATE)
        c = eqx.debug.backward_nan(c, "[quadratic roots] c", terminate=TERMINATE)

    # 1: Check for complex roots.
    discriminant = b**2 - 4 * a * c
    is_complex = discriminant <= 0

    if DBG:
        discriminant = eqx.debug.backward_nan(discriminant, "discriminant")

    # gradient is infinity at 0, so clip to small eps.
    discr_safe = jnp.where(is_complex, 1e-6, discriminant)

    if DBG:
        discr_safe = eqx.debug.backward_nan(discr_safe, "sqrt_discr_safe")

    sqrt_discr = jnp.sqrt(discr_safe)

    if DBG:
        sqrt_discr = eqx.debug.backward_nan(sqrt_discr, "sqrt_discr")

    # 2: Compute the roots.
    safe_denom_2a = jnp.where(jnp.abs(a) < 1e-6, 1.0, 2 * a)
    if DBG:
        safe_denom_2a = eqx.debug.backward_nan(safe_denom_2a, "[quadratic_roots] safe_denom_2a")

    denom_x2_bpos = -b - sqrt_discr
    denom_x1_bneg = -b + sqrt_discr

    if DBG:
        denom_x2_bpos = eqx.debug.backward_nan(denom_x2_bpos, "[quadratic_roots] denom_x2_bpos")
        denom_x1_bneg = eqx.debug.backward_nan(denom_x1_bneg, "[quadratic_roots] denom_x1_bneg")

    safe_denom_x2_bpos = jnp.where(jnp.abs(denom_x2_bpos) < 1e-6, 1.0, denom_x2_bpos)
    safe_denom_x1_bneg = jnp.where(jnp.abs(denom_x1_bneg) < 1e-6, 1.0, denom_x1_bneg)

    # safe_denom_x2_bpos = eqx.debug.backward_nan(safe_denom_x2_bpos, "[quadratic_roots] safe_denom_x2_bpos")
    # safe_denom_x1_bneg = eqx.debug.backward_nan(safe_denom_x1_bneg, "[quadratic_roots] safe_denom_x1_bneg")

    x1_bpos = (-b - sqrt_discr) / safe_denom_2a
    x2_bpos = (2 * c) / safe_denom_x2_bpos

    x1_bneg = (2 * c) / safe_denom_x1_bneg
    x2_bneg = (-b + sqrt_discr) / safe_denom_2a

    # x1_bpos = eqx.debug.backward_nan(x1_bpos, "[quadratic_roots] x1_bpos")
    # x2_bpos = eqx.debug.backward_nan(x2_bpos, "[quadratic_roots] x2_bpos")

    x1 = jnp.where(b >= 0, x1_bpos, x1_bneg)
    x2 = jnp.where(b >= 0, x2_bpos, x2_bneg)

    # Make sure x1 and x2 are not nan.
    is_valid = (~is_complex) & jnp.isfinite(x1) & jnp.isfinite(x2)

    # x1 = eqx.debug.backward_nan(x1, "x1")
    # x2 = eqx.debug.backward_nan(x2, "x2")

    return x1, x2, is_valid
