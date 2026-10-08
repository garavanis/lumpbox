"""Simulations checked against closed-form results."""

from functools import partial

import numpy as np
import pytest
import scipy.signal

import lumpbox as lb

# an SDOF oscillator, x(0) = 0 and xdot(0) = v0
M, K, V0 = 1.0, 100.0, 1.0
WN = np.sqrt(K / M)
TIME = np.linspace(0.0, 2.0, 2001)  # dt = 1 ms, about 3 periods

SOLVERS = {
    "RK4": lb.RK4,
    "RK45": partial(lb.RK45, rtol=1e-10, atol=1e-12),
}


def _sdof(c, nonlinearity=None):
    return lb.MDOFCantilever(
        np.array([M]), np.array([c]), np.array([K]), nonlinearity=nonlinearity
    )


@pytest.mark.parametrize("solver", SOLVERS.values(), ids=SOLVERS.keys())
@pytest.mark.parametrize("zeta", [0.0, 0.05])
def test_sdof_free_vibration(solver, zeta):
    system = _sdof(c=2 * zeta * WN * M)
    data = system.simulate(TIME, z0=np.array([0.0, V0]), solver=solver)

    wd = WN * np.sqrt(1 - zeta**2)
    x = V0 / wd * np.exp(-zeta * WN * TIME) * np.sin(wd * TIME)
    np.testing.assert_allclose(data["x"][0], x, atol=1e-6)
    # the default observation, the acceleration, follows the equation of motion
    np.testing.assert_allclose(
        data["acc"][0], -(system.C @ data["xdot"] + system.K @ data["x"])[0], atol=1e-9
    )


@pytest.mark.parametrize(
    "build",
    [
        # one cubic spring to the ground
        lambda kn: _sdof(0.0, lb.ExponentStiffness(np.array([kn]))),
        # a cubic spring to each wall; the restoring force is (kn + kn) x^3
        lambda kn: lb.MDOFSymmetric(
            np.array([M]),
            np.zeros(2),
            np.array([K / 2, K / 2]),
            nonlinearity=lb.ExponentStiffness(np.array([kn / 2, kn / 2]), boundary="fixed"),
        ),
    ],
    ids=["cantilever", "symmetric"],
)
def test_duffing_conserves_energy(build):
    kn = 1e4
    system = build(kn)
    data = system.simulate(TIME, z0=np.array([0.1, 0.0]))

    x, v = data["x"][0], data["xdot"][0]
    energy = 0.5 * M * v**2 + 0.5 * K * x**2 + 0.25 * kn * x**4
    np.testing.assert_allclose(energy, energy[0], rtol=1e-6)


def test_modal_analysis_fixed_fixed_chain():
    # a uniform fixed-fixed chain: w_j = 2 sqrt(k/m) sin(j pi / (2 (n + 1)));
    # C = (c/k) K, so the modes stay classical with zeta_j = (c/k) w_j / 2
    n, m, c, k = 5, 2.0, 0.5, 1e3
    system = lb.MDOFSymmetric(m, c, k, dofs=n)

    j = np.arange(1, n + 1)
    wn = 2 * np.sqrt(k / m) * np.sin(j * np.pi / (2 * (n + 1)))
    zeta = (c / k) * wn / 2

    undamped = system.modal_analysis()
    np.testing.assert_allclose(undamped["wn"], wn)
    phi = undamped["modes"]
    np.testing.assert_allclose(phi.T @ system.M @ phi, np.eye(n), atol=1e-12)

    damped = system.modal_analysis(damped=True)
    np.testing.assert_allclose(damped["fn"], wn / (2 * np.pi))
    np.testing.assert_allclose(damped["zeta"], zeta)
    np.testing.assert_allclose(damped["fd"], wn * np.sqrt(1 - zeta**2) / (2 * np.pi))


def test_batch_matches_single_systems():
    n, fs = 3, 200.0
    time = np.arange(0, 4096) / fs
    systems = [
        lb.MDOFSymmetric(m, 0.5, 1e3, dofs=n, S_a=np.eye(n)[[0, 2]])
        for m in (1.0, 1.5, 2.0)
    ]
    excitations = [lb.BandedNoise((1.0, 40.0), amplitude=1.0)] + [None] * (n - 1)
    F = np.stack(
        [lb.MDOFShaker(excitations, seed=p).generate(time) for p in range(len(systems))]
    )

    single = []
    for system, Fp in zip(systems, F):
        system.excitations = list(Fp)
        single.append(system.simulate(time)["acc"])
    single = np.stack(single)

    solver_batch, obs = lb.stack_systems(systems)
    np.testing.assert_allclose(lb.vmapped_acc(solver_batch, obs, F, time), single, atol=1e-10)

    # chunk_size=2 pads the population of 3 to 4
    freqs, psd = lb.get_psd_data(solver_batch, obs, F, time, fs=fs, nperseg=512, chunk_size=2)
    freqs_ref, psd_ref = scipy.signal.welch(single, fs=fs, nperseg=512, axis=-1)
    np.testing.assert_allclose(freqs, freqs_ref)
    np.testing.assert_allclose(psd, psd_ref, rtol=1e-8, atol=1e-14)
