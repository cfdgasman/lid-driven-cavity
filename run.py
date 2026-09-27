"""Solve the Re = 100 cavity, compare with Ghia et al. (1982) and write figures to docs/."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cavity import divergence, ghia, solve


def ghia_errors(r):
    y, u = r.centreline_u()
    x, v = r.centreline_v()
    eu = np.interp(ghia.Y, y, u) - ghia.U_RE100
    ev = np.interp(ghia.X, x, v) - ghia.V_RE100
    return np.abs(eu).max(), np.abs(ev).max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--study", action="store_true", help="also run a grid-refinement study")
    args = parser.parse_args()

    t0 = time.perf_counter()
    r = solve(args.n, 100.0)
    print(f"N={args.n}: {r.iterations} steps, residual {r.residual:.1e}, "
          f"max|div u| {np.abs(divergence(r)).max():.1e}, {time.perf_counter() - t0:.1f} s")
    print("max |error| vs Ghia: u = %.4f, v = %.4f" % ghia_errors(r))

    # --- centreline validation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    y, u = r.centreline_u()
    a1.plot(u, y, "-", lw=2, label=f"This solver ({args.n}×{args.n})")
    a1.plot(ghia.U_RE100, ghia.Y, "o", mfc="none", ms=7, label="Ghia et al. (1982)")
    a1.set(xlabel="u", ylabel="y", title="u along x = 0.5")
    x, v = r.centreline_v()
    a2.plot(x, v, "-", lw=2, label=f"This solver ({args.n}×{args.n})")
    a2.plot(ghia.X, ghia.V_RE100, "o", mfc="none", ms=7, label="Ghia et al. (1982)")
    a2.set(xlabel="x", ylabel="v", title="v along y = 0.5")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Lid-driven cavity, Re = 100")
    fig.tight_layout()
    fig.savefig("docs/validation_re100.png", dpi=130)

    # --- streamlines
    psi = r.streamfunction()
    xc = np.linspace(0, 1, args.n + 1)
    uc, vc = r.cell_centre_velocity()
    speed = np.hypot(uc, vc)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    im = ax.pcolormesh(xc, xc, speed.T, cmap="viridis", shading="flat")
    levels = np.concatenate((np.linspace(psi.min(), 0, 16)[:-1], [-1e-5, 1e-6, 1e-5]))
    ax.contour(xc, xc, psi.T, levels=np.sort(levels), colors="white", linewidths=0.7, linestyles="solid")
    fig.colorbar(im, ax=ax, label="|u|")
    ax.set(aspect="equal", xlabel="x", ylabel="y", title="Streamlines, Re = 100")
    fig.tight_layout()
    fig.savefig("docs/streamlines_re100.png", dpi=130)

    if args.study:
        print("\n| N | max err u | max err v | psi_min |\n|---|---|---|---|")
        psis = []
        for n in (16, 32, 64, 128):
            rr = r if n == args.n else solve(n, 100.0)
            eu, ev = ghia_errors(rr)
            psis.append(-rr.streamfunction().min())
            print(f"| {n} | {eu:.4f} | {ev:.4f} | {-psis[-1]:.6f} |")
        f1, f2, f3 = psis[-3:]
        order = np.log2((f1 - f2) / (f2 - f3))
        extrap = f3 + (f3 - f2) / (2**order - 1)
        print(f"\nobserved order of |psi_min| (N = 32/64/128): {order:.2f}")
        print(f"Richardson-extrapolated psi_min: {-extrap:.6f}  (Ghia et al.: -0.103423)")


if __name__ == "__main__":
    main()
