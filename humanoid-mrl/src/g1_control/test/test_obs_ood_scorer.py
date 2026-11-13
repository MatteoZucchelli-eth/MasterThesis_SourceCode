# MIT License
"""Unit tests for the out-of-distribution scorer."""

import numpy as np
import pytest


@pytest.fixture
def scorer(tmp_path):
    """An ObsOODScorer trained on a small synthetic Gaussian dataset."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    from g1_control.utils.obs_ood_scorer import ObsOODScorer

    rng = np.random.default_rng(0)
    dim = 5
    states = rng.standard_normal((400, dim)).astype(np.float32)

    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"observation.state": list(states)})
    df.to_parquet(data_dir / "episode_000.parquet")

    return ObsOODScorer(str(tmp_path), k=20, sigma=1.0)


def test_score_returns_expected_keys(scorer):
    out = scorer.score(np.zeros(5, dtype=np.float32))
    assert set(out) == {"energy", "mean_knn_dist"}
    assert np.isfinite(out["energy"])
    assert out["mean_knn_dist"] >= 0.0


def test_in_distribution_has_lower_energy_than_far_outlier(scorer):
    near = scorer.score(np.zeros(5, dtype=np.float32))["energy"]
    far = scorer.score(np.full(5, 50.0, dtype=np.float32))["energy"]
    assert far > near


def test_energy_and_grad_matches_finite_difference(scorer):
    s = np.array([0.3, -0.2, 0.1, 0.0, 0.4], dtype=np.float32)
    energy, grad = scorer.energy_and_grad(s)
    assert np.isfinite(energy)
    assert grad.shape == s.shape

    eps = 1e-3
    fd = np.zeros_like(s)
    for i in range(s.shape[0]):
        d = np.zeros_like(s)
        d[i] = eps
        e_plus, _ = scorer.energy_and_grad(s + d)
        e_minus, _ = scorer.energy_and_grad(s - d)
        fd[i] = (e_plus - e_minus) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
