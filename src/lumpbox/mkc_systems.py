import warnings
import numpy as np
import scipy.linalg
from lumpbox.exc import MDOFShaker
from lumpbox.solvers import RK4


class StateSpaceSystem:
    """Base class for any state-space represented system."""

    def gen_state_matrices(self):
        """State matrices A, H, and nonlinear coupling An."""
        M_inv = np.linalg.inv(self.M)
        self.A = np.block(
            [
                [np.zeros((self.dofs, self.dofs)), np.eye(self.dofs)],
                [-M_inv @ self.K, -M_inv @ self.C],
            ]
        )
        self.H = np.concatenate((np.zeros((self.dofs, self.dofs)), M_inv), axis=0)
        self.An = self._build_nonlinear_block(
            M_inv, n_zero_rows=self.dofs, selection=np.eye(self.dofs)
        )

        if self.nonlinearity is not None:
            coupling = self.Kn if self.Kn is not None else self.Cn
            cols = coupling.shape[1]
            Drel = np.zeros((cols, self.dofs))
            Drel[: self.dofs, : self.dofs] = np.eye(self.dofs) - np.eye(self.dofs, k=-1)
            if cols == self.dofs + 1:
                Drel[self.dofs, self.dofs - 1] = 1.0
            self.Drel = Drel
        else:
            self.Drel = None

    def gen_obs_matrices(self):
        """Observation matrices B, D, and nonlinear coupling Bn."""
        M_inv = np.linalg.inv(self.M)
        dofs = self.dofs
        B_blocks, D_blocks = [], []
        n_zero_rows = 0

        if self.S_d is not None:
            n_d = self.S_d.shape[
                0
            ]  # example of S_d: np.eye(dofs)[[0, 2], :] for dofs=3 and measuring displacements of first and third masses
            B_blocks.append(np.concatenate((self.S_d, np.zeros((n_d, dofs))), axis=1))
            D_blocks.append(np.zeros((n_d, dofs)))
            n_zero_rows += n_d

        if self.S_v is not None:
            n_v = self.S_v.shape[0]
            B_blocks.append(np.concatenate((np.zeros((n_v, dofs)), self.S_v), axis=1))
            D_blocks.append(np.zeros((n_v, dofs)))
            n_zero_rows += n_v

        if self.S_a is not None:
            B_blocks.append(
                np.concatenate(
                    (-self.S_a @ M_inv @ self.K, -self.S_a @ M_inv @ self.C),
                    axis=1,
                )
            )
            D_blocks.append(self.S_a @ M_inv)

        self.B = np.concatenate(B_blocks, axis=0)
        self.D = np.concatenate(D_blocks, axis=0)
        self.Bn = self._build_nonlinear_block(
            M_inv, n_zero_rows=n_zero_rows, selection=self.S_a
        )

    def _build_nonlinear_block(self, M_inv, n_zero_rows, selection):
        """Assemble [-S M^-1 Kn | -S M^-1 Cn] with n_zero_rows zeros above."""
        if selection is None:
            return None

        match [self.Kn, self.Cn]:
            case [None, None]:
                return None
            case [_, None]:
                cols = self.Kn.shape[1]
                Kn_part = -selection @ M_inv @ self.Kn
                Cn_part = np.zeros((selection.shape[0], cols))
            case [None, _]:
                cols = self.Cn.shape[1]
                Kn_part = np.zeros((selection.shape[0], cols))
                Cn_part = -selection @ M_inv @ self.Cn
            case [_, _]:
                cols = self.Kn.shape[1]
                Kn_part = -selection @ M_inv @ self.Kn
                Cn_part = -selection @ M_inv @ self.Cn

        bottom = np.concatenate((Kn_part, Cn_part), axis=1)
        if n_zero_rows > 0:
            return np.concatenate((np.zeros((n_zero_rows, 2 * cols)), bottom), axis=0)
        return bottom


class MDOFSystem(StateSpaceSystem):
    """
    Generic MDOF system with optional nonlinear coupling. S_d, S_v, S_a are
    (n, dofs) row-selection matrices, e.g. np.eye(dofs)[[0, 2], :].
    """

    def __init__(
        self,
        M,
        C,
        K,
        Cn=None,
        Kn=None,
        nonlinearity=None,
        S_d=None,
        S_v=None,
        S_a=None,
    ):
        self.M = M
        self.C = C
        self.K = K
        self.Cn = Cn
        self.Kn = Kn
        self.dofs = M.shape[0]
        self.nonlinearity = nonlinearity

        # if no selection matrices provided, default to measuring all accelerations
        if S_d is None and S_v is None and S_a is None:
            S_a = np.eye(self.dofs)
        self.S_d = S_d
        self.S_v = S_v
        self.S_a = S_a

        self.gen_state_matrices()
        self.gen_obs_matrices()

    def modal_analysis(self, damped=False):
        """
        Linear modal analysis. Operates on the linear M, C, K only; any
        nonlinearity is ignored (this is a linearization about equilibrium).

        Undamped (damped=False): solves the generalized eigenvalue problem
        K phi = omega^2 M phi via the symmetric solver. Returns:
            fn    natural frequencies [Hz], ascending
            wn    angular natural frequencies [rad/s]
            modes mass-normalized mode shapes (columns match fn order)

        Damped (damped=True): eigenvalues of the state matrix A. Returns:
            fd    damped natural frequencies [Hz]
            fn    undamped natural frequencies [Hz]
            zeta  modal damping ratios
        """
        if not damped:
            w2, phi = scipy.linalg.eigh(self.K, self.M)  # ascending eigenvalues
            wn = np.sqrt(np.clip(w2, 0.0, None))  # rad/s
            return {"fn": wn / (2 * np.pi), "wn": wn, "modes": phi}

        lam = np.linalg.eigvals(self.A)  # complex conjugate pairs
        pos = lam[lam.imag > 0]  # keep one eigenvalue per pair
        pos = pos[np.argsort(np.abs(pos))]
        wn = np.abs(pos)
        return {
            "fd": pos.imag / (2 * np.pi),
            "fn": wn / (2 * np.pi),
            "zeta": -pos.real / wn,
        }

    def _split_observations(self, y):
        """Split stacked output y into named slices."""
        slices = {}
        row = 0
        if self.S_d is not None:
            n_d = self.S_d.shape[0]
            slices["disp"] = y[row : row + n_d]
            row += n_d
        if self.S_v is not None:
            n_v = self.S_v.shape[0]
            slices["vel"] = y[row : row + n_v]
            row += n_v
        if self.S_a is not None:
            n_a = self.S_a.shape[0]
            slices["acc"] = y[row : row + n_a]
        return slices

    def simulate(self, tt, z0=None, solver=None):
        """Simulate the system over time samples tt."""
        self.t = tt

        if hasattr(self, "actuator"):
            self.f = self.actuator.generate(tt)
        elif hasattr(self, "excitations"):
            self.shaker = MDOFShaker(self.excitations)
            self.f = self.shaker.generate(tt)
        else:
            self.f = None

        self.solver = RK4(self) if solver is None else solver(self)

        if z0 is None:
            z0 = np.zeros(2 * self.dofs)
            if hasattr(self, "excitations") and all(
                e is None for e in self.excitations
            ):
                warnings.warn(
                    "Zero initial condition and zero excitations, what do you want??",
                    UserWarning,
                )

        simulated_state_data = self.solver.sim(tt, z0, self.f)

        z = simulated_state_data["z"]
        if self.solver.__class__.__name__ in ("RK4", "RK45"):
            y = self.B @ z
            if self.f is not None:
                y = y + self.D @ self.f
            if self.Bn is not None:
                y = y + self.Bn @ self.solver.nonlin_input(z)
            simulated_state_data.update(self._split_observations(y))

        return simulated_state_data


class MDOFSymmetric(MDOFSystem):
    """
    Generic MDOF symmetric system. Both ends fixed: `dofs` masses are coupled
    by `dofs + 1` springs/dampers, so `c_` and `k_` (when given as arrays)
    must have length `dofs + 1`. `m_` has length `dofs`.
    """

    def __init__(
        self,
        m_,
        c_,
        k_,
        dofs=None,
        nonlinearity=None,
        S_d=None,
        S_v=None,
        S_a=None,
    ):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
            if c_.shape[0] != dofs + 1 or k_.shape[0] != dofs + 1:
                raise ValueError(
                    f"Symmetric system with {dofs} DOFs needs c_ and k_ of "
                    f"length {dofs + 1}, got {c_.shape[0]} and {k_.shape[0]}"
                )
        elif dofs is not None:
            m_ = m_ * np.ones(dofs)
            c_ = c_ * np.ones(dofs + 1)
            k_ = k_ * np.ones(dofs + 1)
        else:
            raise Exception(
                "Under defined system, please provide either parameter vectors or number of degrees of freedom"
            )

        self.m_ = m_
        self.c_ = c_
        self.k_ = k_

        M = np.diag(m_)
        C = (
            np.diag(c_[:-1] + c_[1:])
            + np.diag(-c_[1:-1], k=1)
            + np.diag(-c_[1:-1], k=-1)
        )
        K = (
            np.diag(k_[:-1] + k_[1:])
            + np.diag(-k_[1:-1], k=1)
            + np.diag(-k_[1:-1], k=-1)
        )

        if nonlinearity is not None:
            super().__init__(
                M,
                C,
                K,
                Cn=nonlinearity.Cn,
                Kn=nonlinearity.Kn,
                nonlinearity=nonlinearity,
                S_d=S_d,
                S_v=S_v,
                S_a=S_a,
            )
        else:
            super().__init__(M, C, K, S_d=S_d, S_v=S_v, S_a=S_a)


class MDOFCantilever(MDOFSystem):
    """
    Generic MDOF "cantilever" system. Fixed at the base, free at the top:
    `dofs` masses coupled by `dofs` springs/dampers, so `m_`, `c_`, `k_`
    all have length `dofs`.
    """

    def __init__(
        self,
        m_,
        c_,
        k_,
        dofs=None,
        nonlinearity=None,
        S_d=None,
        S_v=None,
        S_a=None,
    ):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * np.ones(dofs)
            c_ = c_ * np.ones(dofs)
            k_ = k_ * np.ones(dofs)
        else:
            raise Exception(
                "Under defined system, please provide either parameter vectors or number of degrees of freedom"
            )

        self.m_ = m_
        self.c_ = c_
        self.k_ = k_

        M = np.diag(m_)
        C = (
            np.diag(np.concatenate((c_[:-1] + c_[1:], np.array([c_[-1]])), axis=0))
            + np.diag(-c_[1:], k=1)
            + np.diag(-c_[1:], k=-1)
        )
        K = (
            np.diag(np.concatenate((k_[:-1] + k_[1:], np.array([k_[-1]])), axis=0))
            + np.diag(-k_[1:], k=1)
            + np.diag(-k_[1:], k=-1)
        )

        if nonlinearity is not None:
            super().__init__(
                M,
                C,
                K,
                Cn=nonlinearity.Cn,
                Kn=nonlinearity.Kn,
                nonlinearity=nonlinearity,
                S_d=S_d,
                S_v=S_v,
                S_a=S_a,
            )
        else:
            super().__init__(M, C, K, S_d=S_d, S_v=S_v, S_a=S_a)
