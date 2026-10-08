import numpy as np


def sample_normal(mean, cov, n_samples, rng):
    """Draw Gaussian samples for each material property.

    The standard deviation follows the coefficient of variation,
    ``std = cov * mean``, so each entry is drawn from ``N(mean, cov * mean)``.

    Parameters
    ----------
    mean : array_like
        Mean value per material, shape ``(n_materials,)``.
    cov : array_like
        Coefficient of variation per material, shape ``(n_materials,)``.
    n_samples : int
        Number of samples to draw per material.
    rng : numpy.random.Generator
        Random generator used to draw the samples.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(n_samples, n_materials)``.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_1d(np.asarray(cov, dtype=float))
    std = cov * mean
    return rng.normal(loc=mean, scale=std, size=(n_samples, mean.size))


def temperature():
    pass


def wind_speed():
    pass
