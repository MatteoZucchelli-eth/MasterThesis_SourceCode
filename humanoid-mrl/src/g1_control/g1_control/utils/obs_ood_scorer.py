from __future__ import annotations

import glob

import logging

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class ObsOODScorer:
    """Scores how out-of-distribution a current observation is vs training data."""

    def __init__(
        self,
        dataset_path: str,
        k: int = 100,
        sigma: float = 0.3733,
    ) -> None:
        logger.info("ObsOODScorer: loading state data from '%s' ...", dataset_path)
        states = self._load_states(dataset_path)
        self._k = k
        self._sigma = sigma

        # Per-dimension whitening: the state mixes metres, rot6d and radians.
        self._mean = states.mean(axis=0)
        self._std = states.std(axis=0)
        self._std[self._std < 1e-6] = 1.0
        states = (states - self._mean) / self._std

        self._state_tree = KDTree(states)

        recommended_sigma = self._median_knn_sigma(states)
        logger.info(
            "ObsOODScorer: ready — %d training frames, k=%s, sigma=%s "
            "(recommended: %.4f)",
            len(states),
            k,
            sigma,
            recommended_sigma,
        )

    # ── public ────────────────────────────────────────────────────────────────

    def score(
        self,
        obs_state: np.ndarray,
    ) -> dict:
        """Score the current observation."""
        energy, dists, _ = self._energy_and_grad(obs_state)
        result = {"energy": energy, "mean_knn_dist": float(np.mean(dists))}

        return result

    def grad(self, obs_state: np.ndarray) -> np.ndarray:
        """Gradient of the energy E = -log p(s) with respect to s."""
        _, _, grad = self._energy_and_grad(obs_state)
        return grad

    def energy_and_grad(self, obs_state: np.ndarray) -> tuple[float, np.ndarray]:
        """Energy and its gradient wrt s from a single k-NN query."""
        energy, _, grad = self._energy_and_grad(obs_state)
        return energy, grad

    # ── private ───────────────────────────────────────────────────────────────

    def _energy_and_grad(
        self, obs_state: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Shared core: energy E = -log p(s) and ∂E/∂s from one k-NN query.

        Everything is computed in whitened space z = (s - mean) / std. The
        gradient is mapped back to the original state units via the chain rule
        ∂E/∂s = (∂E/∂z) / std. Returns (energy, knn distances in z-space, grad).

        Gaussian KDE p(z) = (1/k) Σ exp(-‖z-nᵢ‖²/2σ²); energy via log-sum-exp.
        ∂E/∂z = (1/σ²) Σ wᵢ (z - nᵢ) with wᵢ softmax weights over the exponents.
        """
        z = (obs_state - self._mean) / self._std
        dists, idxs = self._state_tree.query(z[np.newaxis], k=self._k)
        # SciPy squeezes k=1 results; force (k,) shape.
        dists = np.atleast_1d(dists[0])  # (k,)
        idxs = np.atleast_1d(idxs[0])  # (k,)
        neighbors = self._state_tree.data[idxs]  # (k, d)

        exponents = -(dists**2) / (2 * self._sigma**2)
        max_exp = np.max(exponents)
        log_p = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))

        weights = np.exp(exponents - max_exp)
        weights /= weights.sum()  # (k,) normalized

        grad_z = (
            np.sum(weights[:, np.newaxis] * (z - neighbors), axis=0) / self._sigma**2
        )
        return float(-log_p), dists, grad_z / self._std

    def _median_knn_sigma(self, states: np.ndarray, sample_size: int = 10_000) -> float:
        """Estimate σ as the median k-th NN distance in the training data.

        Queries k+1 neighbours (the first is the point itself). Query points
        are subsampled to bound startup; the tree is complete.
        """
        if len(states) > sample_size:
            rng = np.random.default_rng(0)
            states = states[rng.choice(len(states), size=sample_size, replace=False)]
        dists, _ = self._state_tree.query(states, k=self._k + 1)
        return float(np.median(dists[:, -1]))

    def _load_states(self, dataset_path: str) -> np.ndarray:
        parquet_files = sorted(
            glob.glob(f"{dataset_path}/data/**/*.parquet", recursive=True)
        )
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under '{dataset_path}/data/'. "
                "Check the dataset_path parameter."
            )
        frames = [
            np.stack(
                pd.read_parquet(f, columns=["observation.state"])[
                    "observation.state"
                ].to_numpy()
            )
            for f in parquet_files
        ]
        return np.concatenate(frames, axis=0).astype(np.float32)
