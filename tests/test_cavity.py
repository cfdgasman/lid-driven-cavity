import numpy as np
import pytest

from cavity import divergence, ghia, solve


@pytest.fixture(scope="module")
def result():
    return solve(n=32, re=100.0, tol=1e-5)


def test_converges(result):
    assert result.residual < 1e-5


def test_discretely_divergence_free(result):
    assert np.abs(divergence(result)).max() < 1e-9


def test_matches_ghia_re100(result):
    y, u = result.centreline_u()
    x, v = result.centreline_v()
    assert np.abs(np.interp(ghia.Y, y, u) - ghia.U_RE100).max() < 0.02
    assert np.abs(np.interp(ghia.X, x, v) - ghia.V_RE100).max() < 0.02


def test_odd_grid_rejected():
    with pytest.raises(ValueError):
        solve(n=31)
