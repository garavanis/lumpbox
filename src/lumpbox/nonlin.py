import numpy as np
import jax.numpy as jnp


def _build_coupling(coefs, boundary):
    """Build the nonlinear coupling matrix for an MDOF chain.

    boundary='free': coefs has length n. Returns (n, n) Fixed-Free matrix:
        [κ_1, -κ_2,  0,   ..., 0  ]
        [0,    κ_2, -κ_3, ..., 0  ]
        [...                      ]
        [0,    0,    ...,   0, κ_n]

    boundary='fixed': coefs has length n+1. Returns (n, n+1) Fixed-Fixed
    matrix that accounts for the trailing grounded spring κ_{n+1}:
        [κ_1, -κ_2,  0,   ..., 0,   0      ]
        [0,    κ_2, -κ_3, ..., 0,   0      ]
        [0,      0,  κ_3, ..., 0,   0      ]
        [...                               ]
        [0,    0,    0,   ..., κ_n, κ_{n+1}]
    The bottom-right entry is +κ_{n+1} because the (n+1)-th component of
    g(u) is u_n^3 (i.e. signed so that adjacent and grounded contributions
    on mass n both enter the EoM with positive sign).
    """
    coefs = np.asarray(coefs)
    if boundary == "free":
        return np.diag(coefs) - np.diag(coefs[1:], 1)
    if boundary == "fixed":
        n = coefs.shape[0] - 1
        M = np.zeros((n, n + 1))
        M[np.arange(n), np.arange(n)] = coefs[:n]
        if n > 1:
            M[np.arange(n - 1), np.arange(1, n)] = -coefs[1:n]
        M[n - 1, n] = coefs[n]
        return M
    raise ValueError(f"Unknown boundary {boundary!r}; use 'free' or 'fixed'.")


class Nonlinearity:

    def __init__(self, dofs, boundary="free"):

        self.dofs = dofs
        if self.dofs is not None:
            cols = dofs if boundary == "free" else dofs + 1
            self.Cn = np.zeros((self.dofs, cols))
            self.Kn = np.zeros((self.dofs, cols))

    def gc_func(self, x, xdot):
        return jnp.zeros_like(xdot)

    def gk_func(self, x, xdot):
        return jnp.zeros_like(x)


class ExponentStiffness(Nonlinearity):

    def __init__(self, kn_, exponent=3, dofs=None, boundary="free"):
        self.exponent = exponent
        self.boundary = boundary
        match kn_:
            case np.ndarray():
                self.kn_ = kn_
                dofs = kn_.shape[0] if boundary == "free" else kn_.shape[0] - 1
            case None:
                raise ValueError(
                    "kn_ must be provided as an ndarray or scalar; got None."
                )
            case _:
                if dofs is None:
                    raise ValueError("dofs must be provided when kn_ is a scalar.")
                size = dofs if boundary == "free" else dofs + 1
                self.kn_ = kn_ * np.ones(size)
        super().__init__(dofs, boundary)
        self.Kn = _build_coupling(self.kn_, boundary)

    def gk_func(self, x, xdot):
        return jnp.sign(x) * jnp.abs(x) ** self.exponent


class ExponentDamping(Nonlinearity):

    def __init__(self, cn_, exponent=0.5, dofs=None, boundary="free"):
        self.exponent = exponent
        self.boundary = boundary
        match cn_:
            case np.ndarray():
                self.cn_ = cn_
                dofs = cn_.shape[0] if boundary == "free" else cn_.shape[0] - 1
            case None:
                raise ValueError(
                    "cn_ must be provided as an ndarray or scalar; got None."
                )
            case _:
                if dofs is None:
                    raise ValueError("dofs must be provided when cn_ is a scalar.")
                size = dofs if boundary == "free" else dofs + 1
                self.cn_ = cn_ * np.ones(size)
        super().__init__(dofs, boundary)
        self.Cn = _build_coupling(self.cn_, boundary)

    def gc_func(self, x, xdot):
        return jnp.sign(xdot) * jnp.abs(xdot) ** self.exponent


class VanDerPol(Nonlinearity):

    def __init__(self, cn_, dofs=None, boundary="free"):
        self.boundary = boundary
        match cn_:
            case np.ndarray():
                self.cn_ = cn_
                dofs = cn_.shape[0] if boundary == "free" else cn_.shape[0] - 1
            case None:
                raise ValueError(
                    "cn_ must be provided as an ndarray or scalar; got None."
                )
            case _:
                if dofs is None:
                    raise ValueError("dofs must be provided when cn_ is a scalar.")
                size = dofs if boundary == "free" else dofs + 1
                self.cn_ = cn_ * np.ones(size)
        super().__init__(dofs, boundary)
        self.Cn = _build_coupling(self.cn_, boundary)

    def gc_func(self, x, xdot):
        return (x**2 - 1.0) * xdot
