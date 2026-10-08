from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax


class Solver(eqx.Module):
    """Equinox base for time-domain solvers of state-space systems."""

    A: jax.Array
    H: jax.Array
    An: Optional[jax.Array]
    Drel: Optional[jax.Array]
    dofs: int = eqx.field(static=True)
    gk: Optional[Callable] = eqx.field(static=True)
    gc: Optional[Callable] = eqx.field(static=True)

    def __init__(self, system):
        self.A = jnp.asarray(system.A)
        self.H = jnp.asarray(system.H)
        self.dofs = system.dofs
        if system.An is not None:
            self.An = jnp.asarray(system.An)
            self.Drel = jnp.asarray(system.Drel)
            self.gk = system.nonlinearity.gk_func
            self.gc = system.nonlinearity.gc_func
        else:
            self.An = None
            self.Drel = None
            self.gk = None
            self.gc = None

    def nonlin_input(self, z):
        """Nonlinear coupling input g(z) = [gk(x_rel, xdot_rel); gc(x_rel, xdot_rel)].

        Returns ``None`` for a linear system. Accepts a single state
        ``(2*dofs,)`` (the integrator's per-step use) or a full trajectory
        ``(2*dofs, nt)`` (the observation path's use), since ``Drel`` and the
        elementwise kernels broadcast over the trailing time axis.
        """
        if self.An is None:
            return None
        x, xdot = z[: self.dofs], z[self.dofs :]
        x_rel, xdot_rel = self.Drel @ x, self.Drel @ xdot
        return jnp.concatenate(
            (self.gk(x_rel, xdot_rel), self.gc(x_rel, xdot_rel)), axis=0
        )

    def rhs(self, z, f):
        """State derivative ż = A z (+ An g(z)) (+ H f)."""
        out = self.A @ z
        if self.An is not None:
            out = out + self.An @ self.nonlin_input(z)
        if f is not None:
            out = out + self.H @ f
        return out

    def _pack(self, z):
        """Pack an integrated trajectory (2*dofs, nt) into the result dict."""
        return {"x": z[: self.dofs, :], "xdot": z[self.dofs :, :], "z": z}

    def sim(self, time, z0, F=None):
        raise NotImplementedError


class RK4(Solver):
    """Fixed-step classical Runge-Kutta (4th order) via jax.lax.scan."""

    def sim(self, time, z0, F=None):
        z0j = jnp.asarray(z0, dtype=jnp.float64)  # type promotion safeguard
        dt = time[1] - time[0]

        if F is not None:
            F = jnp.asarray(F)  # (n_ch, nt)
            F_t = F[:, :-1].T  # (ns-1, n_ch)  start of step
            F_m = (0.5 * (F[:, :-1] + F[:, 1:])).T  # (ns-1, n_ch)  midpoint
            F_e = F[:, 1:].T  # (ns-1, n_ch)  end of step

            def step(z, xs):
                f_t, f_m, f_e = xs
                k1 = self.rhs(z, f_t)
                k2 = self.rhs(z + (dt / 2) * k1, f_m)
                k3 = self.rhs(z + (dt / 2) * k2, f_m)
                k4 = self.rhs(z + dt * k3, f_e)
                z1 = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                return z1, z1  # carry, output

            _, z_steps = lax.scan(step, z0j, (F_t, F_m, F_e))
        else:

            def step(z, _):
                k1 = self.rhs(z, None)
                k2 = self.rhs(z + (dt / 2) * k1, None)
                k3 = self.rhs(z + (dt / 2) * k2, None)
                k4 = self.rhs(z + dt * k3, None)
                z1 = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                return z1, z1

            _, z_steps = lax.scan(step, z0j, None, length=len(time) - 1)

        # z_steps: (ns-1, 2*dofs) — prepend z0, transpose to (2*dofs, ns)
        z_full = jnp.concatenate([z0j[None], z_steps], axis=0).T
        return self._pack(z_full)


class RK45(Solver):
    """Adaptive Dormand-Prince 5(4) solver via diffrax."""

    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)

    def __init__(self, system, rtol=1e-3, atol=1e-6):
        super().__init__(system)
        self.rtol = rtol
        self.atol = atol

    def sim(self, time, z0, F=None):
        t_jax = jnp.asarray(time, dtype=jnp.float64)
        z0j = jnp.asarray(z0, dtype=jnp.float64)

        if F is not None:
            F = jnp.asarray(F, dtype=jnp.float64)  # (n_ch, nt)
            force_path = diffrax.LinearInterpolation(ts=t_jax, ys=F.T)

            def vf(t, y, args):
                return self.rhs(y, force_path.evaluate(t))

        else:

            def vf(t, y, args):
                return self.rhs(y, None)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            diffrax.Dopri5(),
            t0=t_jax[0],
            t1=t_jax[-1],
            dt0=t_jax[1] - t_jax[0],
            y0=z0j,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            saveat=diffrax.SaveAt(ts=t_jax),
            max_steps=16 * len(time),
        )

        z_full = sol.ys.T  # (2*dofs, nt)
        return self._pack(z_full)
