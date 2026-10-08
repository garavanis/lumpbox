"""lumpbox: lumped-mass (mass-spring-damper, MKC) systems toybox, built on JAX.

The main names are available here (``import lumpbox as lb``); the modules hold
the rest.
"""

from importlib import metadata as _metadata

import jax

# the solvers integrate in double precision; set before any module builds arrays
jax.config.update("jax_enable_x64", True)

try:
    __version__ = _metadata.version("lumpbox")
except _metadata.PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0+unknown"

from lumpbox import (  # noqa: E402
    batchsolve,
    exc,
    mkc_systems,
    nonlin,
    obs_models,
    solvers,
)
from lumpbox.batchsolve import get_psd_data, stack_systems, vmapped_acc  # noqa: E402
from lumpbox.exc import (  # noqa: E402
    BandedNoise,
    Impulse,
    MDOFShaker,
    RandPhaseMS,
    SineSweep,
    Sinusoid,
    WhiteGaussian,
)
from lumpbox.mkc_systems import MDOFCantilever, MDOFSymmetric, MDOFSystem  # noqa: E402
from lumpbox.nonlin import ExponentDamping, ExponentStiffness, VanDerPol  # noqa: E402
from lumpbox.obs_models import (  # noqa: E402
    CompositeObservation,
    GaussianNoiseObservation,
    IdentityObservation,
    LinearObservation,
)
from lumpbox.solvers import RK4, RK45  # noqa: E402

__all__ = [
    # systems
    "MDOFSystem",
    "MDOFSymmetric",
    "MDOFCantilever",
    # nonlinearities
    "ExponentStiffness",
    "ExponentDamping",
    "VanDerPol",
    # excitations
    "Impulse",
    "Sinusoid",
    "WhiteGaussian",
    "BandedNoise",
    "SineSweep",
    "RandPhaseMS",
    "MDOFShaker",
    # solvers
    "RK4",
    "RK45",
    # batched simulation
    "stack_systems",
    "vmapped_acc",
    "get_psd_data",
    # observation models
    "IdentityObservation",
    "LinearObservation",
    "GaussianNoiseObservation",
    "CompositeObservation",
    # modules
    "batchsolve",
    "exc",
    "mkc_systems",
    "nonlin",
    "obs_models",
    "solvers",
]
