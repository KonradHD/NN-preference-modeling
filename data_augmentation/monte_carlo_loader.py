"""
Loader for the Monte Carlo generated dataset.

Directory layout (as produced by MonteCarloDataGenerator):
  data/monte-carlo/
    criteria{n}/
      {phase}/              <- train / validation / test
        low_inconsistency/   matrices.npy  weights.npy
        moderate_inconsistency/
        high_inconsistency/

Usage
-----
  loader = MonteCarloLoader()                          # default path
  loader = MonteCarloLoader("data/monte-carlo")        # explicit path

  # Load all strata for train, criteria=5
  matrices, weights = loader.load(Phase.TRAIN, num_criteria=5)

  # Load only specific strata
  matrices, weights = loader.load(
      Phase.TRAIN, num_criteria=5,
      strata=["low_inconsistency", "moderate_inconsistency"]
  )

  # Limit the number of matrices per stratum
  matrices, weights = loader.load(Phase.TRAIN, num_criteria=5, max_per_stratum=10_000)
"""

import os
import glob
import warnings
import numpy as np

from utils.phase import Phase


STRATA_NAMES = [
    "low_inconsistency",
    "moderate_inconsistency",
    "high_inconsistency",
]


class MonteCarloLoader:
    """
    Loads pre-generated Monte Carlo AHP matrices from disk.

    Parameters
    ----------
    base_dir : str | None
        Root directory of the Monte Carlo dataset.
        Defaults to ``data/monte-carlo``.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or os.path.join("data", "monte-carlo")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        phase: Phase,
        num_criteria: int,
        strata: list[str] | None = None,
        max_per_stratum: int | None = None,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all (or selected) strata for a given phase and criteria size.

        Parameters
        ----------
        phase : Phase
            TRAIN, VALIDATION, or TEST.
        num_criteria : int
            Matrix size n (3-9).
        strata : list[str] | None
            Names of strata to include.  If None, all three strata are loaded.
            Valid values: ``"low_inconsistency"``, ``"moderate_inconsistency"``,
            ``"high_inconsistency"``.
        max_per_stratum : int | None
            Maximum number of matrices to load from each stratum.
            If None, all available matrices are loaded.
        alpha : float | None
            If given, load only the weights file matching this Dirichlet alpha
            (e.g. alpha=2.0 -> ``weights_alpha2_00.npy``).
        sigma : float | None
            If given, load only the matrices file matching this sigma
            (e.g. sigma=0.9 -> ``matrices_sigma0_90.npy``).

        Returns
        -------
        matrices : np.ndarray, shape (N, n, n)
        weights  : np.ndarray, shape (N, n)
        """
        strata = strata or STRATA_NAMES

        all_matrices: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []

        for stratum in strata:
            m, w = self._load_stratum(
                phase=phase,
                num_criteria=num_criteria,
                stratum=stratum,
                max_count=max_per_stratum,
                alpha=alpha,
                sigma=sigma,
            )
            all_matrices.append(m)
            all_weights.append(w)

        matrices = np.concatenate(all_matrices, axis=0)
        weights = np.concatenate(all_weights, axis=0)

        # Shuffle to avoid phase-aligned ordering
        idx = np.random.permutation(len(matrices))
        return matrices[idx], weights[idx]

    def load_stratum(
        self,
        phase: Phase,
        num_criteria: int,
        stratum: str,
        max_count: int | None = None,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a single stratum directly (public convenience method).

        Parameters
        ----------
        alpha : float | None
            If given, look for the weights file generated with this exact
            Dirichlet alpha (e.g. ``weights_alpha2_00.npy``).
        sigma : float | None
            If given, look for the matrices file generated with this exact
            sigma (e.g. ``matrices_sigma0_90.npy``).
        """
        return self._load_stratum(phase, num_criteria, stratum, max_count,
                                  alpha=alpha, sigma=sigma)

    def dataset_info(self, num_criteria: int) -> dict:
        """
        Return a summary of available matrices per phase and stratum.

        Returns a nested dict: {phase_name: {stratum_name: count}}.
        """
        info: dict[str, dict[str, int]] = {}
        for phase in Phase:
            info[phase.value] = {}
            for stratum in STRATA_NAMES:
                path = self._stratum_dir(phase, num_criteria, stratum)
                mpath = self._find_file(path, "matrices")
                if mpath is not None:
                    arr = np.load(mpath, mmap_mode="r")
                    info[phase.value][stratum] = len(arr)
                else:
                    info[phase.value][stratum] = 0
        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stratum_dir(self, phase: Phase, num_criteria: int, stratum: str) -> str:
        return os.path.join(
            self.base_dir,
            f"criteria{num_criteria}",
            phase.value,
            stratum,
        )

    @staticmethod
    def _float_to_token(value: float) -> str:
        """Convert a float to the underscore-delimited token used in filenames.

        Examples: 2.0 -> '2_00', 0.9 -> '0_90', 1.30 -> '1_30'
        """
        return f"{value:.2f}".replace(".", "_")

    def _find_file(self, directory: str, prefix: str,
                   exact_suffix: str | None = None) -> str | None:
        """Find a .npy file for *prefix* in *directory*.

        If *exact_suffix* is given (e.g. ``'sigma0_90'``), look for
        ``{prefix}_{exact_suffix}.npy`` specifically.  Otherwise use a glob
        to find any ``{prefix}_*.npy`` and fall back to ``{prefix}.npy``.

        Returns None if no matching file exists.
        """
        if exact_suffix is not None:
            exact_path = os.path.join(directory, f"{prefix}_{exact_suffix}.npy")
            if os.path.isfile(exact_path):
                return exact_path
            warnings.warn(
                f"Exact file '{exact_path}' not found; falling back to glob search."
            )

        # Glob: matrices_sigma*.npy / weights_alpha*.npy
        pattern = os.path.join(directory, f"{prefix}_*.npy")
        matches = sorted(glob.glob(pattern))
        if matches:
            if len(matches) > 1:
                warnings.warn(
                    f"Multiple files matching '{pattern}' found; using '{matches[-1]}'. "
                    "Delete stale files to avoid ambiguity."
                )
            return matches[-1]

        # Legacy fallback: matrices.npy / weights.npy
        legacy = os.path.join(directory, f"{prefix}.npy")
        if os.path.isfile(legacy):
            return legacy
        return None

    def _load_stratum(
        self,
        phase: Phase,
        num_criteria: int,
        stratum: str,
        max_count: int | None,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if stratum not in STRATA_NAMES:
            raise ValueError(
                f"Unknown stratum '{stratum}'. Valid: {STRATA_NAMES}"
            )

        save_dir = self._stratum_dir(phase, num_criteria, stratum)

        sigma_suffix = f"sigma{self._float_to_token(sigma)}" if sigma is not None else None
        alpha_suffix = f"alpha{self._float_to_token(alpha)}" if alpha is not None else None

        matrices_path = self._find_file(save_dir, "matrices", exact_suffix=sigma_suffix)
        weights_path  = self._find_file(save_dir, "weights",  exact_suffix=alpha_suffix)

        if matrices_path is None or weights_path is None:
            raise FileNotFoundError(
                f"Monte Carlo data not found at '{save_dir}'. "
                "Run generations.py first."
            )

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)

        if max_count is not None and len(matrices) > max_count:
            matrices = matrices[:max_count]
            weights = weights[:max_count]
        elif max_count is not None and len(matrices) < max_count:
            warnings.warn(
                f"Requested {max_count} matrices from stratum '{stratum}' "
                f"but only {len(matrices)} are available."
            )

        return matrices, weights
