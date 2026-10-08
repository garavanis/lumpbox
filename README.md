# lumpbox

Lumped-mass (mass-spring-damper) system simulation in Python, built on JAX:
chains of masses, linear or nonlinear, driven by shakers and observed through
sensors, simulated one at a time or as a whole population at once.

> In development: the interface may still change.

The systems follow

```math
\mathbf{M}\ddot{\mathbf{x}} + \mathbf{C}\dot{\mathbf{x}} + \mathbf{K}\mathbf{x}
+ \mathbf{K}_n g_k(\Delta\mathbf{x}, \Delta\dot{\mathbf{x}})
+ \mathbf{C}_n g_c(\Delta\mathbf{x}, \Delta\dot{\mathbf{x}}) = \mathbf{f}
```

where $\Delta\mathbf{x}$ are the extensions of the springs, and the nonlinear
forces $g_k$, $g_c$ act on them (a cubic spring has $g_k = \Delta x^3$). The
systems are integrated in state-space form, $\mathbf{z} = [\mathbf{x};
\dot{\mathbf{x}}]$, and observed through selection matrices $\mathbf{S}_d$,
$\mathbf{S}_v$, $\mathbf{S}_a$ that pick the measured displacements,
velocities and accelerations.

- **Systems**: chains fixed at both ends (`MDOFSymmetric`) or at the base only
  (`MDOFCantilever`), or any `M`, `C`, `K` (`MDOFSystem`); linear modal analysis.
- **Nonlinearities**: exponent stiffness (e.g. Duffing), exponent damping, Van der Pol.
- **Excitations**: impulse, sinusoid, white and band-limited noise, sine sweep,
  random-phase multisine, or any force array; one independent random stream per DOF.
- **Solvers**: fixed-step RK4 (`jax.lax.scan`) and adaptive Dormand-Prince
  (`diffrax`).
- **Populations**: many systems stacked and simulated together with
  `vmap`; the PSDs are computed in chunks to bound the memory.
- **Sensors**: gain and offset, Gaussian noise at a given SNR.

Built on [JAX](https://github.com/jax-ml/jax),
[Equinox](https://github.com/patrick-kidger/equinox) and
[Diffrax](https://github.com/patrick-kidger/diffrax). Importing `lumpbox`
switches JAX to 64-bit precision.

## Install

The repository is private; with access to it:

```bash
uv add git+https://github.com/garavanis/lumpbox
# or
pip install git+https://github.com/garavanis/lumpbox
```

## Quick start

A 3-DOF chain fixed at both ends, with a cubic spring between masses 1 and 2,
and band-limited noise on mass 1:

```python
import numpy as np
import lumpbox as lb

n_dof = 3
fs = 100.0
time = np.arange(10_000) / fs

# 3 masses joined by 4 springs and dampers; a cubic stiffness on the second spring
kn = np.array([0.0, 1e9, 0.0, 0.0])
system = lb.MDOFSymmetric(
    np.ones(n_dof), 4.0 * np.ones(n_dof + 1), 1e3 * np.ones(n_dof + 1),
    nonlinearity=lb.ExponentStiffness(kn, exponent=3, boundary="fixed"),
    S_d=np.eye(n_dof), S_a=np.eye(n_dof),   # measure all displacements and accelerations
)
system.excitations = [lb.BandedNoise(bandwidth=(0.0, 50.0), amplitude=1.0), None, None]

data = system.simulate(time)                # RK4; solver=lb.RK45 for adaptive steps
data["x"], data["xdot"]                     # states, (n_dof, nt)
data["disp"], data["acc"]                   # observations, (n_obs, nt)

acc = lb.GaussianNoiseObservation(snr=20).observe(data["acc"])   # noisy sensors
system.modal_analysis()["fn"]               # natural frequencies of the linear part [Hz]
```

A population, simulated together (accelerations only):

```python
systems = [
    lb.MDOFSymmetric(m, 4.0, 1e3, dofs=n_dof, S_a=np.eye(n_dof))
    for m in np.linspace(0.9, 1.1, 100)
]
solver_batch, obs = lb.stack_systems(systems)

excitations = [lb.BandedNoise((0.0, 50.0), amplitude=1.0), None, None]
F = np.stack([lb.MDOFShaker(excitations, seed=p).generate(time) for p in range(len(systems))])

acc = lb.vmapped_acc(solver_batch, obs, F, time)        # (P, n_a, nt)
freqs, psd = lb.get_psd_data(solver_batch, obs, F, time, fs=fs, nperseg=1024)
```

## Modules

The systems, nonlinearities, excitations, solvers and sensors are available
at the top level (`import lumpbox as lb`); the modules hold everything:

| Module | Contents |
| --- | --- |
| `mkc_systems` | `MDOFSystem`, `MDOFSymmetric`, `MDOFCantilever`: state-space matrices, simulation, modal analysis |
| `nonlin` | `ExponentStiffness`, `ExponentDamping`, `VanDerPol` |
| `exc` | the excitations, and `MDOFShaker`, which assembles them per DOF |
| `solvers` | `RK4`, `RK45` (Equinox modules) |
| `batchsolve` | `stack_systems`, `vmapped_acc`, `get_psd_data`: populations |
| `obs_models` | sensor models: identity, gain and offset, Gaussian noise, and their composition |
| `latent_effects` | sampling of uncertain material properties |

## Tests

```bash
uv run pytest
```

## Acknowledgements

lumpbox started from [dynasim](https://github.com/MarcusHA94/dynasim) by
Marcus Haywood-Alexander.

## License

MIT, see [LICENSE](LICENSE).
