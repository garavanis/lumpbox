from abc import ABC, abstractmethod

import numpy as np
import scipy.signal


class Excitation(ABC):

    @abstractmethod
    def generate(self, time, seed=None):
        """Generate the force time-series sampled at ``time``.

        ``MDOFShaker`` uses ``seed`` to draw an independent stream per
        (excitation, member, DOF).
        """
        pass


class Impulse(Excitation):
    """
    Impulse excitation
    """

    def __init__(self, imp_times=None, width=0.1, f0=0.1):
        self.imp_times = [0] if imp_times is None else imp_times
        self.width = width
        self.f0 = f0

    def generate(self, time, seed=None):
        ns = time.shape[0]
        nt_width = int(self.width / (time[1] - time[0])) // 2
        f = np.zeros(ns)
        imp_locs = [np.argmin(np.abs(time - imp_time)) for imp_time in self.imp_times]
        for imp_loc in imp_locs:
            f[imp_loc - nt_width : imp_loc + nt_width] = self.f0
        return f


class Sinusoid(Excitation):
    """
    Single sinusoidal signal with central frequency w, amplitude f0, and phase phi
    """

    def __init__(self, w, f0=1.0, phi=0):
        self.w = w
        self.f0 = f0
        self.phi = phi

    def generate(self, time, seed=None):
        return self.f0 * np.sin(self.w * time + self.phi)


class WhiteGaussian(Excitation):
    """
    White Gaussian noise with variance f0, and mean
    """

    def __init__(self, f0, mean=0.0, seed=42):
        self.f0 = f0
        self.u = mean
        self.seed = seed

    def generate(self, time, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        return rng.normal(self.u, self.f0, size=time.shape[0])


class BandedNoise(Excitation):

    def __init__(self, bandwidth, amplitude, seed=42):
        self.bandwidth = bandwidth
        self.amplitude = amplitude
        self.seed = seed

    @staticmethod
    def _fftnoise(f, rng):
        f = np.array(f, dtype="complex")
        Np = (len(f) - 1) // 2
        phases = rng.random(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1 : Np + 1] *= phases
        f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
        return np.fft.ifft(f).real

    @staticmethod
    def _band_limited_noise(min_freq, max_freq, samples, samplerate, rng):
        freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        f[idx] = 1
        return BandedNoise._fftnoise(f, rng)

    def generate(self, time, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        dt = time[1] - time[0]
        noise = self._band_limited_noise(
            self.bandwidth[0],
            self.bandwidth[1],
            samples=len(time),
            samplerate=1 / dt,
            rng=rng,
        )
        noise = noise - noise.mean()
        scale = np.std(noise)
        if scale == 0:
            raise ValueError(
                f"Band {self.bandwidth} contains no representable frequencies for dt={dt}"
            )
        return self.amplitude * noise / scale


class SineSweep(Excitation):
    """
    Sine sweep signal
    """

    def __init__(self, w_l, w_u, F0=1.0, scale="linear"):
        self.w_l = w_l
        self.w_u = w_u
        self.F0 = F0
        self.scale = scale

    def generate(self, time, seed=None):
        f0 = self.w_l / (2 * np.pi)
        f1 = self.w_u / (2 * np.pi)
        return self.F0 * scipy.signal.chirp(time, f0, time[-1], f1, method=self.scale)


class RandPhaseMS(Excitation):
    """
    Random-phase multi-sine
    """

    def __init__(self, freqs, Sx, seed=42):
        self.freqs = freqs
        self.Sx = Sx
        self.seed = seed

    def generate(self, time, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        phases = rng.standard_normal(self.freqs.shape[0]) * np.pi / 2
        F_mat = np.sin(time.reshape(-1, 1) @ self.freqs.reshape(1, -1) + phases.T)
        return (F_mat @ self.Sx).reshape(-1)


class MDOFShaker:
    """
    MDOF shaker: assembles a per-DOF force matrix from a list of excitations
    (``Excitation`` instances, raw ``np.ndarray`` forces, or ``None`` for a
    silent DOF).

    Seeding contract. A stochastic excitation on DOF ``n`` is drawn from an
    independent stream derived with

        np.random.SeedSequence([excitation.seed, shaker.seed, n])
    """

    def __init__(self, excitations=None, seed=0):
        self.excitations = excitations
        self.dofs = len(excitations)
        self.seed = seed

    def generate(self, time):
        nt = time.shape[0]
        self.f = np.zeros((self.dofs, nt))
        offset = 0 if self.seed is None else self.seed
        for n, excite in enumerate(self.excitations):
            match excite:
                case Excitation():
                    base = getattr(excite, "seed", 0)
                    ss = np.random.SeedSequence([base, offset, n])
                    self.f[n, :] = excite.generate(time, seed=ss)
                case np.ndarray():
                    self.f[n, :] = excite
                case None:
                    pass
        return self.f
