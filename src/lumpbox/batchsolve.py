"""Batched, vmapped simulation pipeline for population-scale dataset generation."""

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from lumpbox.solvers import RK4


# batched solver + batched obs arrays
def stack_systems(systems, *, solver=RK4, **solver_kw):
    """Stack pre-built systems into one batched Equinox solver + obs arrays.
    The pipeline observes only accelerations.
    """
    if len(systems) == 0:
        raise ValueError("population needs at least one system")

    solvers, Ba, Da, Bna = [], [], [], []
    for p, sysp in enumerate(systems):
        if sysp.S_a is None:
            raise ValueError(
                f"systems[{p}] must have S_a set (acceleration-only pipeline)"
            )
        if sysp.S_d is not None or sysp.S_v is not None:
            raise ValueError(
                f"systems[{p}] has S_d/S_v set; this pipeline observes only "
                "accelerations — build each system with S_a only."
            )
        solvers.append(solver(sysp, **solver_kw))
        Ba.append(sysp.B)
        Da.append(sysp.D)
        Bna.append(sysp.Bn)

    solver_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *solvers)
    obs = {
        "Ba": jnp.asarray(np.stack(Ba)),
        "Da": jnp.asarray(np.stack(Da)),
        "Bna": None if Bna[0] is None else jnp.asarray(np.stack(Bna)),
    }
    return solver_batch, obs


@eqx.filter_jit
def vmapped_acc(solver_batch, obs, F, time, z0=None):
    """vmapped acceleration trajectory, ``(P, n_a, nt)``.

    Integrates each member's state, then forms
    ``y = Ba z + Da F (+ Bna g(z))`` per member.
    """
    time = jnp.asarray(time, dtype=jnp.float64)
    F = jnp.asarray(F)
    n2 = solver_batch.A.shape[-1]
    z0j = (
        jnp.zeros(n2, dtype=jnp.float64)
        if z0 is None
        else jnp.asarray(z0, dtype=jnp.float64)
    )

    def one(solver, ob, Fi):
        z = solver.sim(time, z0j, Fi)["z"]
        y = ob["Ba"] @ z + ob["Da"] @ Fi
        if ob["Bna"] is not None:
            y = y + ob["Bna"] @ solver.nonlin_input(z)
        return y

    return eqx.filter_vmap(one)(solver_batch, obs, F)


# bound device memory for large populations
def get_psd_data(
    solver_batch,
    obs,
    F,
    time,
    *,
    fs,
    nperseg,
    noverlap=None,
    chunk_size=25,
    z0=None,
):
    """Chunked ``acc → psd``. Only the PSD lands on the host.

    Returns
    -------
    (freqs, psd) : (n_f,), (P, n_a, n_f)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    P = F.shape[0]
    n_pad = (-P) % chunk_size

    def pad(x):
        x = jnp.asarray(x)
        if n_pad == 0:
            return x
        widths = [(0, n_pad)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, widths)

    sb_p = jax.tree_util.tree_map(pad, solver_batch)
    obs_p = jax.tree_util.tree_map(pad, obs)
    F_p = pad(F)

    # compile once, reuse
    jitted_welch = eqx.filter_jit(jax.scipy.signal.welch)

    chunks = []
    freqs = None
    for start in range(0, P + n_pad, chunk_size):
        sl = slice(start, start + chunk_size)
        sb = jax.tree_util.tree_map(lambda x: x[sl], sb_p)
        ob = jax.tree_util.tree_map(lambda x: x[sl], obs_p)
        a_chunk = vmapped_acc(sb, ob, F_p[sl], time, z0=z0)
        freqs, p_chunk = jitted_welch(
            a_chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1
        )
        chunks.append(np.asarray(p_chunk))

    return np.asarray(freqs), np.concatenate(chunks, axis=0)[:P]
