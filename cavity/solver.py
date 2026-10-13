"""2D incompressible Navier-Stokes solver for the lid-driven cavity.

Discretisation
--------------
* Finite-volume, staggered (MAC) grid on the unit square with N x N cells.
  u lives on vertical faces, v on horizontal faces, p at cell centres.
* Convection: second-order central, conservative (flux) form.
* Diffusion: second-order central.
* Time integration: explicit Euler for momentum, Chorin projection for
  the pressure-velocity coupling. The pressure Poisson equation (pure
  Neumann) is solved with a sparse LU factorisation computed once.
* The lid moves with u = U at y = 1. Other walls are no-slip. Wall
  values on the staggered grid are enforced through ghost cells.

Array layout (index [i, j] = [x, y]):
    u : (N+1, N+2)  faces i = 0..N, cell rows j = 0..N+1 (0, N+1 are ghosts)
    v : (N+2, N+1)  cell columns i = 0..N+1 (ghosts), faces j = 0..N
    p : (N, N)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


@dataclass
class Result:
    n: int
    re: float
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    iterations: int
    residual: float

    @property
    def h(self) -> float:
        return 1.0 / self.n

    def centreline_u(self):
        """u(y) along the vertical centreline x = 0.5 (requires even N)."""
        i = self.n // 2
        y = (np.arange(self.n) + 0.5) * self.h
        return np.concatenate(([0.0], y, [1.0])), np.concatenate(
            ([0.0], self.u[i, 1:-1], [1.0])
        )

    def centreline_v(self):
        """v(x) along the horizontal centreline y = 0.5 (requires even N)."""
        j = self.n // 2
        x = (np.arange(self.n) + 0.5) * self.h
        return np.concatenate(([0.0], x, [1.0])), np.concatenate(
            ([0.0], self.v[1:-1, j], [0.0])
        )

    def cell_centre_velocity(self):
        uc = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        vc = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        return uc, vc

    def streamfunction(self):
        """psi on cell corners, from integrating u = dpsi/dy up from psi = 0 at y = 0."""
        psi = np.zeros((self.n + 1, self.n + 1))
        psi[:, 1:] = np.cumsum(self.u[:, 1:-1] * self.h, axis=1)
        return psi


def _neumann_laplacian(n: int, h: float):
    """5-point Laplacian with homogeneous Neumann BCs; one value pinned to remove the null space."""
    main = 2.0 * np.ones(n)
    main[0] = main[-1] = 1.0
    d1 = sp.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1])
    eye = sp.identity(n)
    a = -(sp.kron(d1, eye) + sp.kron(eye, d1)) / h**2
    a = a.tolil()
    a[0, :] = 0.0
    a[0, 0] = 1.0
    return splu(a.tocsc())


def _apply_bcs(u, v, lid):
    u[:, 0] = -u[:, 1]  # bottom wall, u = 0
    u[:, -1] = 2.0 * lid - u[:, -2]  # moving lid, u = lid
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[0, :] = -v[1, :]  # left wall, v = 0
    v[-1, :] = -v[-2, :]  # right wall, v = 0
    v[:, 0] = 0.0
    v[:, -1] = 0.0


def solve(
    n: int = 64,
    re: float = 100.0,
    lid: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100_000,
    cfl: float = 0.8,
) -> Result:
    """March to steady state and return the converged fields.

    Convergence: max |u^{n+1} - u^n| / dt < tol (a steady-state residual).
    """
    if n % 2:
        raise ValueError("use an even number of cells so the centrelines lie on faces")
    h = 1.0 / n
    nu = lid / re  # L = 1, so Re = U L / nu
    dt = cfl * min(0.25 * h * h / nu, h / abs(lid))

    u = np.zeros((n + 1, n + 2))
    v = np.zeros((n + 2, n + 1))
    p = np.zeros((n, n))
    _apply_bcs(u, v, lid)
    poisson = _neumann_laplacian(n, h)

    residual = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        # --- u-momentum on interior u faces: i = 1..n-1, j = 1..n
        ue = 0.5 * (u[2:, 1:-1] + u[1:-1, 1:-1])
        uw = 0.5 * (u[1:-1, 1:-1] + u[:-2, 1:-1])
        un = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
        us = 0.5 * (u[1:-1, :-2] + u[1:-1, 1:-1])
        vn = 0.5 * (v[1:-2, 1:] + v[2:-1, 1:])
        vs = 0.5 * (v[1:-2, :-1] + v[2:-1, :-1])
        conv_u = (ue * ue - uw * uw) / h + (un * vn - us * vs) / h
        lap_u = (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        ) / h**2
        u_star = u.copy()
        u_star[1:-1, 1:-1] += dt * (-conv_u + nu * lap_u)

        # --- v-momentum on interior v faces: i = 1..n, j = 1..n-1
        ve = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])
        vw = 0.5 * (v[:-2, 1:-1] + v[1:-1, 1:-1])
        vn2 = 0.5 * (v[1:-1, 1:-1] + v[1:-1, 2:])
        vs2 = 0.5 * (v[1:-1, :-2] + v[1:-1, 1:-1])
        ue2 = 0.5 * (u[1:, 1:-2] + u[1:, 2:-1])
        uw2 = 0.5 * (u[:-1, 1:-2] + u[:-1, 2:-1])
        conv_v = (ue2 * ve - uw2 * vw) / h + (vn2 * vn2 - vs2 * vs2) / h
        lap_v = (
            v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4.0 * v[1:-1, 1:-1]
        ) / h**2
        v_star = v.copy()
        v_star[1:-1, 1:-1] += dt * (-conv_v + nu * lap_v)

        # --- pressure Poisson: lap(p) = div(u*) / dt
        div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / h + (
            v_star[1:-1, 1:] - v_star[1:-1, :-1]
        ) / h
        rhs = (div / dt).ravel()
        rhs[0] = 0.0
        p = poisson.solve(rhs).reshape(n, n)

        # --- projection
        u_new = u_star
        v_new = v_star
        u_new[1:-1, 1:-1] -= dt * (p[1:, :] - p[:-1, :]) / h
        v_new[1:-1, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / h
        _apply_bcs(u_new, v_new, lid)

        residual = max(np.abs(u_new - u).max(), np.abs(v_new - v).max()) / dt
        u, v = u_new, v_new
        if residual < tol:
            break

    return Result(n=n, re=re, u=u, v=v, p=p - p.mean(), iterations=it, residual=residual)


def divergence(result: Result) -> np.ndarray:
    """Discrete divergence in every cell; should be ~machine precision."""
    h = result.h
    return (result.u[1:, 1:-1] - result.u[:-1, 1:-1]) / h + (
        result.v[1:-1, 1:] - result.v[1:-1, :-1]
    ) / h
