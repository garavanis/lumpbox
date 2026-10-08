import numpy as np
from abc import ABC, abstractmethod


class ObservationModel(ABC):
    """
    Abstract base class for observation models.

    Observation models transform a true vibration signal (e.g.,
    displacement, velocity, or acceleration response at one or more DOFs)
    into a measured signal. Transformations may include sensor gain/offset,
    measurement noise, or other channel effects.
    """

    @abstractmethod
    def observe(self, x, t=None):
        """
        Transform a true signal into a measured signal.

        Parameters
        ----------
        x : float or array
            True signal value(s) at one or more DOFs.
        t : float or array, optional
            Time of observation.

        Returns
        -------
        float or array
            Measured signal.
        """
        pass


class IdentityObservation(ObservationModel):
    """
    Identity observation model that returns the signal unchanged.
    """

    def observe(self, x, t=None):
        """
        Return the signal unchanged.

        Parameters
        ----------
        x : float or array
            True signal value(s) at one or more DOFs.
        t : float or array, optional
            Time of observation (not used).

        Returns
        -------
        float or array
            Same as input signal.
        """
        return x


class LinearObservation(ObservationModel):
    """
    Linear observation model with scaling and offset.

    Implements ``y = scale * x + offset``, e.g., sensor gain and bias.
    """

    def __init__(self, scale=1.0, offset=0.0):
        """
        Initialize with scaling factor and offset.

        Parameters
        ----------
        scale : float, optional
            Sensor gain applied to the signal.
        offset : float, optional
            Bias added after scaling.
        """
        self.scale = scale
        self.offset = offset

    def observe(self, x, t=None):
        """
        Apply the linear transformation to the signal.

        Parameters
        ----------
        x : float or array
            True signal value(s) at one or more DOFs.
        t : float or array, optional
            Time of observation (not used).

        Returns
        -------
        float or array
            Transformed signal: ``scale * x + offset``.
        """
        return self.scale * x + self.offset


class GaussianNoiseObservation(ObservationModel):
    """
    Add Gaussian noise to a signal. The noise level is specified in one of
    three mutually exclusive ways:

    - ``std_dev``: explicit standard deviation, identical for every DOF.
    - ``db``: absolute noise power in dB; noise variance is
      ``10**(db/10)``, identical for every DOF.
    - ``snr``: signal-to-noise ratio in dB; noise variance is chosen
      per-DOF so that ``10*log10(P_sig / P_noise) == snr``.

    The signal can be a scalar, a 1-D array of shape ``(ns,)``, or a 2-D
    array of shape ``(nd, ns)``.
    """

    def __init__(self, std_dev=None, db=None, snr=None, seed=42):
        """
        Initialize with one of the three noise-level specifications.

        Parameters
        ----------
        std_dev : float, optional
            Standard deviation of the Gaussian noise.
        db : float, optional
            Absolute noise power in dB.
        snr : float, optional
            Signal-to-noise ratio in dB, computed per-DOF.
        seed : int, optional
            Random seed. Each DOF uses ``seed + i``.

        Raises
        ------
        ValueError
            If none, or more than one, of ``std_dev``, ``db``, ``snr`` are
            provided.
        """
        specified = sum(p is not None for p in (std_dev, db, snr))
        if specified == 0:
            raise ValueError("Specify one of `std_dev`, `db`, or `snr`.")
        if specified > 1:
            raise ValueError("Specify only one of `std_dev`, `db`, or `snr`.")
        self.std_dev = std_dev
        self.db = db
        self.snr = snr
        self.seed = seed

    def observe(self, x, t=None):
        """
        Add Gaussian noise to the signal.

        Parameters
        ----------
        x : float or array
            Signal, scalar or of shape ``(ns,)`` or ``(nd, ns)``.
        t : float or array, optional
            Time of observation (not used).

        Returns
        -------
        float or array
            Signal with added noise, same shape as ``x``. DOFs with zero
            power are returned unchanged when in SNR mode.
        """
        x_arr = np.asarray(x, dtype=float)
        ndim = x_arr.ndim
        if ndim == 0:
            x_arr = x_arr.reshape(1, 1)
        elif ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        nd, ns = x_arr.shape
        x_noisy = np.empty_like(x_arr)
        for i in range(nd):
            rng = np.random.default_rng(self.seed + i)
            if self.std_dev is not None:
                sigma = self.std_dev
            elif self.db is not None:
                sigma = np.sqrt(10.0 ** (self.db / 10.0))
            else:
                p_sig = np.mean(x_arr[i] ** 2)
                if p_sig == 0.0:
                    x_noisy[i] = x_arr[i]
                    continue
                sigma = np.sqrt(10.0 ** ((10.0 * np.log10(p_sig) - self.snr) / 10.0))
            x_noisy[i] = x_arr[i] + rng.normal(0.0, sigma, size=ns)
        if ndim == 0:
            return x_noisy[0, 0]
        if ndim == 1:
            return x_noisy[0]
        return x_noisy


class CompositeObservation(ObservationModel):
    """
    Combines multiple observation models in sequence.
    """

    def __init__(self, observation_models):
        """
        Initialize with a list of observation models.

        Parameters
        ----------
        observation_models : list
            List of observation models to apply in sequence
        """
        self.models = observation_models

    def observe(self, x, t=None):
        """
        Apply all observation models in sequence.

        Parameters
        ----------
        x : float or array
            True signal value(s) at one or more DOFs.
        t : float or array, optional
            Time of observation.

        Returns
        -------
        float or array
            Signal after all transformations.
        """
        result = x
        for model in self.models:
            result = model.observe(result, t)
        return result
