import numpy as np


# Saaty's fundamental scale: 1, 2, ..., 9 and their reciprocals
SAATY_SCALE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
# All possible values: reciprocals + scale
SAATY_VALUES = np.concatenate([1.0 / SAATY_SCALE[::-1][:-1], SAATY_SCALE])  # 1/9, 1/8, ..., 1, 2, ..., 9


class MonteCarloGenerator:
    """
    Generates AHP pairwise comparison matrices using the Monte Carlo method.

    Pipeline (per batch):
      1. Sample priority weight vectors from a Dirichlet distribution.
      2. Build perfectly consistent comparison matrices from those weights.
      3. Apply multiplicative log-normal noise to upper-triangle entries.
      4. Set diagonal to 1 and fill lower triangle as reciprocals.
      5. Discretize the noisy matrix to the nearest value in Saaty's scale.

    Parameters
    ----------
    num_criteria : int
        Size n of the n×n comparison matrix (must be in 3..9).
    alpha : float or list[float]
        Concentration parameter(s) of the Dirichlet distribution.
        Scalar broadcasts to all n dimensions.
        All ones → uniform distribution on the simplex.
        Values > 1 concentrate weights around 1/n (more realistic).
    noise_sigma : float
        Standard deviation of the log-normal noise (σ > 0).
        Small σ → nearly consistent matrices.
        Large σ → highly inconsistent matrices.
    """

    def __init__(
        self,
        num_criteria: int,
        alpha: float | list[float] = 2.0,
        noise_sigma: float = 0.3,
    ):
        if not (3 <= num_criteria <= 9):
            raise ValueError(f"num_criteria must be in [3, 9], got {num_criteria}")
        if noise_sigma <= 0:
            raise ValueError(f"noise_sigma must be positive, got {noise_sigma}")

        self.num_criteria = num_criteria
        self.noise_sigma = noise_sigma

        if isinstance(alpha, (int, float)):
            self.alpha = np.full(num_criteria, float(alpha))
        else:
            alpha = list(alpha)
            if len(alpha) != num_criteria:
                raise ValueError(
                    f"Length of alpha ({len(alpha)}) must equal num_criteria ({num_criteria})"
                )
            self.alpha = np.array(alpha, dtype=float)

        if np.any(self.alpha <= 0):
            raise ValueError("All alpha values must be positive")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, num_matrices: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate *num_matrices* noisy, discretized AHP matrices.

        Returns
        -------
        matrices : np.ndarray, shape (num_matrices, n, n)
            Pairwise comparison matrices with Saaty-scale entries.
        weights  : np.ndarray, shape (num_matrices, n)
            Ground-truth priority weight vectors (sum to 1).
        """
        weights = self._sample_weights(num_matrices)
        coherent = self._build_coherent(weights)
        noisy = self._apply_noise(coherent)
        discrete = self._discretize(noisy)
        return discrete, weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_weights(self, num_matrices: int) -> np.ndarray:
        """Sample weight vectors from Dirichlet(alpha)."""
        return np.random.dirichlet(self.alpha, size=num_matrices)  # (M, n)

    def _build_coherent(self, weights: np.ndarray) -> np.ndarray:
        """
        Build perfectly consistent matrices: a_ij = w_i / w_j.

        weights : (M, n)
        returns : (M, n, n)
        """
        # weights[:, :, None] → (M, n, 1)
        # weights[:, None, :] → (M, 1, n)
        return weights[:, :, None] / weights[:, None, :]  # (M, n, n)

    def _apply_noise(self, coherent: np.ndarray) -> np.ndarray:
        """
        Apply multiplicative log-normal noise to upper-triangle entries.

        a'_ij = a_ij * ε_ij   where ε ~ LogNormal(0, σ)   for i < j
        a'_ji = 1 / a'_ij
        a'_ii = 1
        """
        M, n, _ = coherent.shape
        noisy = np.ones_like(coherent)

        # Sample noise only for upper triangle elements
        num_upper = n * (n - 1) // 2
        epsilon = np.random.lognormal(
            mean=0.0, sigma=self.noise_sigma, size=(M, num_upper)
        )

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                noisy[:, i, j] = coherent[:, i, j] * epsilon[:, idx]
                noisy[:, j, i] = 1.0 / noisy[:, i, j]
                idx += 1

        return noisy

    def _discretize(self, matrices: np.ndarray) -> np.ndarray:
        """
        Round each upper-triangle entry to the nearest value in Saaty's scale
        (1/9, 1/8, …, 1, 2, …, 9). Ties broken upward.
        Lower triangle is set as reciprocal; diagonal stays 1.
        """
        M, n, _ = matrices.shape
        discrete = np.ones_like(matrices)

        log_saaty = np.log(SAATY_VALUES)  # work in log-space for symmetric rounding

        for i in range(n):
            for j in range(i + 1, n):
                vals = matrices[:, i, j]
                log_vals = np.log(np.clip(vals, 1e-9, None))

                # Find nearest Saaty value (ties → higher value = larger index)
                diffs = np.abs(log_vals[:, None] - log_saaty[None, :])  # (M, S)
                best_idx = np.argmax(-diffs, axis=1)  # rightmost min → higher value wins
                # Correct: use argmin but with small right-bias for ties
                # np.argmin returns leftmost; we want rightmost tie → negate and argmax
                best_idx = diffs.shape[1] - 1 - np.argmin(diffs[:, ::-1], axis=1)
                best_saaty = SAATY_VALUES[best_idx]

                discrete[:, i, j] = best_saaty
                discrete[:, j, i] = 1.0 / best_saaty

        return discrete
