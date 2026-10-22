"""
Monte Carlo data generation pipeline.

Generates AHP pairwise comparison matrices stratified by Consistency Ratio (CR):
  - High consistency   (CR ≤ 0.10) → 40 % of data
  - Moderate           (0.10 < CR ≤ 0.33) → 50 % of data
  - Extreme            (CR > 0.33) → 10 % of data

Each stratum is further split into train / validation / test (80 / 10 / 10 %).

Output directory layout
-----------------------
  data/monte-carlo/
    criteria{n}/
      train/
        high_consistency/      matrices.npy   weights.npy
        moderate_consistency/  matrices.npy   weights.npy
        extreme_consistency/   matrices.npy   weights.npy
      validation/   (same sub-structure)
      test/         (same sub-structure)

Usage (from project root)
-------------------------
  python generate_monte_carlo.py
  python generate_monte_carlo.py --total 500000 --criteria 3 4 5
"""

import os
import logging
import time
import numpy as np
from typing import Callable

from analytic_methods.EVM_method import EVM
from data_augmentation.generator.monte_carlo_generator import MonteCarloGenerator
from utils.phase import Phase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CR thresholds & stratum proportions (from the thesis specification)
# ---------------------------------------------------------------------------

CR_STRATA: dict[str, tuple[float, float, float]] = {
    # name: (cr_min_exclusive, cr_max_inclusive, fraction_of_total)
    "high_consistency":     (0.00, 0.10, 0.40),
    "moderate_consistency": (0.10, 0.33, 0.50),
    "extreme_consistency":  (0.33, float("inf"), 0.10),
}

SPLIT_RATIOS: dict[Phase, float] = {
    Phase.TRAIN:      0.80,
    Phase.VALIDATION: 0.10,
    Phase.TEST:       0.10,
}

# ---------------------------------------------------------------------------
# Calibrated noise σ pairs for each matrix size n
# (low_sigma is used to fill high+moderate strata; high_sigma for extreme)
# Values were calibrated empirically so each batch covers its target stratum
# efficiently, minimising rejection waste.
#
# Rule of thumb from the calibration run:
#   low_sigma ≈ 0.7 gives ~40-60 % high + ~40-55 % moderate for most n
#   high_sigma ≈ 1.0-1.1 gives a comfortable amount of extreme (CR > 0.33)
# ---------------------------------------------------------------------------
_SIGMA_LOW: dict[int, float] = {
    3: 0.90, 4: 0.80, 5: 0.75, 6: 0.70, 7: 0.70, 8: 0.65, 9: 0.65,
}
_SIGMA_HIGH: dict[int, float] = {
    3: 1.30, 4: 1.10, 5: 1.00, 6: 0.95, 7: 0.93, 8: 0.90, 9: 0.88,
}


class MonteCarloDataGenerator:
    """
    Orchestrates the full Monte Carlo data generation pipeline.

    Parameters
    ----------
    base_output_dir : str
        Root directory where generated data are stored.
        Defaults to ``data/monte-carlo``.
    alpha : float | list[float]
        Dirichlet concentration parameter(s).  Scalar is broadcast to all
        ``num_criteria`` dimensions.  Values around 2–3 produce realistic
        weight distributions.
    batch_size : int
        Number of matrices generated in a single batch before filtering.
        Larger batches are faster but use more memory.
    seed : int | None
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        base_output_dir: str = os.path.join("data", "monte-carlo"),
        alpha: float | list[float] = 2.0,
        batch_size: int = 50_000,
        seed: int | None = 42,
    ):
        self.base_output_dir = base_output_dir
        self.alpha = alpha
        self.batch_size = batch_size
        self._evm = EVM()

        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        num_criteria: int,
        total_matrices: int,
        sigma_low: float | None = None,
        sigma_high: float | None = None,
    ) -> None:
        """
        Generate *total_matrices* matrices for a given *num_criteria* size,
        stratify them by CR, split into train/val/test, and save to disk.

        Parameters
        ----------
        num_criteria : int
            Size of the AHP matrix (3–9).
        total_matrices : int
            Total number of matrices to generate (before stratified split).
            This equals the final saved count across all phases and strata.
        sigma_low : float | None
            Override the default low-noise sigma for this criteria size.
            If None, the value from ``_SIGMA_LOW`` is used.
        sigma_high : float | None
            Override the default high-noise sigma for this criteria size.
            If None, the value from ``_SIGMA_HIGH`` is used.
        """
        logger.info(
            "Generating %d matrices for criteria=%d", total_matrices, num_criteria
        )
        t0 = time.time()

        # ---- Determine per-stratum needs ----
        needs: dict[str, int] = {
            name: int(total_matrices * frac)
            for name, (_, _, frac) in CR_STRATA.items()
        }
        # Make sure the rounding adds up to total_matrices
        diff = total_matrices - sum(needs.values())
        needs["moderate_consistency"] += diff  # give the remainder to the largest bin

        # ---- Generate using two sigma regimes ----
        #   "low" sigma → efficient coverage of high + moderate strata
        #   "high" sigma → efficient coverage of extreme stratum
        sigma_low  = sigma_low  if sigma_low  is not None else _SIGMA_LOW.get(num_criteria, 0.70)
        sigma_high = sigma_high if sigma_high is not None else _SIGMA_HIGH.get(num_criteria, 1.00)
        logger.info(
            "  sigma_low=%.2f  sigma_high=%.2f", sigma_low, sigma_high
        )

        # Collect pools per stratum
        pools: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
            name: ([], []) for name in CR_STRATA
        }


        # Generate "low-noise" batches until high + moderate are filled
        low_target  = needs["high_consistency"] + needs["moderate_consistency"]
        high_target = needs["extreme_consistency"]

        self._fill_strata(
            pools=pools,
            strata_to_fill=["high_consistency", "moderate_consistency"],
            target_counts={
                "high_consistency": needs["high_consistency"],
                "moderate_consistency": needs["moderate_consistency"],
            },
            num_criteria=num_criteria,
            sigma=sigma_low,
            budget_factor=3.0,  # generate up to 3× needed to fill bins
        )

        self._fill_strata(
            pools=pools,
            strata_to_fill=["extreme_consistency"],
            target_counts={"extreme_consistency": needs["extreme_consistency"]},
            num_criteria=num_criteria,
            sigma=sigma_high,
            budget_factor=5.0,  # extreme is rarer; allow more oversampling
        )

        def _pool_size(name: str) -> int:
            return sum(len(a) for a in pools[name][0])

        logger.info(
            "Pools filled in %.1fs | high=%d mod=%d ext=%d",
            time.time() - t0,
            _pool_size("high_consistency"),
            _pool_size("moderate_consistency"),
            _pool_size("extreme_consistency"),
        )

        # Map each stratum to the sigma regime that was used to generate it
        stratum_sigma: dict[str, float] = {
            "high_consistency":     sigma_low,
            "moderate_consistency": sigma_low,
            "extreme_consistency":  sigma_high,
        }

        # ---- Concatenate, split, save ----
        for stratum_name, (mats_list, wts_list) in pools.items():
            if not mats_list:
                logger.warning("Stratum '%s' is empty!", stratum_name)
                continue
            matrices = np.concatenate(mats_list, axis=0)[: needs[stratum_name]]
            weights  = np.concatenate(wts_list,  axis=0)[: needs[stratum_name]]

            splits = self._split(matrices, weights)
            for phase, (m_split, w_split) in splits.items():
                self._save(
                    matrices=m_split,
                    weights=w_split,
                    num_criteria=num_criteria,
                    phase=phase,
                    stratum_name=stratum_name,
                    sigma=stratum_sigma[stratum_name],
                    alpha=float(np.mean(self.alpha)),
                )

        logger.info(
            "Done for criteria=%d. Total time: %.1fs", num_criteria, time.time() - t0
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_strata(
        self,
        pools: dict[str, tuple[list, list]],
        strata_to_fill: list[str],
        target_counts: dict[str, int],
        num_criteria: int,
        sigma: float,
        budget_factor: float,
    ) -> None:
        """
        Generate batches until every stratum in *strata_to_fill* has at least
        *target_counts[stratum]* entries, or the budget is exhausted.
        """
        gen = MonteCarloGenerator(
            num_criteria=num_criteria,
            alpha=self.alpha,
            noise_sigma=sigma,
        )
        total_needed = sum(target_counts.values())
        budget = int(total_needed * budget_factor)
        generated = 0

        while generated < budget:
            still_needed = {
                name: target_counts[name] - len(pools[name][0])
                for name in strata_to_fill
                if len(pools[name][0]) < target_counts[name]
            }
            if not still_needed:
                break

            batch = min(self.batch_size, budget - generated)
            mats, wts = gen.generate(batch)
            crs = self._evm.compute_batch(mats)[1]

            # Route each matrix to the appropriate stratum
            for name in strata_to_fill:
                if name not in still_needed:
                    continue
                cr_min, cr_max, _ = CR_STRATA[name]
                if cr_max == float("inf"):
                    mask = crs > cr_min
                else:
                    mask = (crs > cr_min) & (crs <= cr_max)

                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue

                needed = target_counts[name] - len(pools[name][0])
                idx = idx[:needed]
                pools[name][0].append(mats[idx])
                pools[name][1].append(wts[idx])

            generated += batch

        for name in strata_to_fill:
            filled = len(pools[name][0])
            if filled == 0:
                logger.warning(
                    "Stratum '%s' could not be filled (sigma=%.2f, budget=%d). "
                    "Try increasing budget_factor or sigma.",
                    name, sigma, budget,
                )
            elif sum(len(a) for a in pools[name][0]) < target_counts[name]:
                actual = sum(len(a) for a in pools[name][0])
                logger.warning(
                    "Stratum '%s': wanted %d, got only %d.",
                    name, target_counts[name], actual,
                )

    def _split(
        self,
        matrices: np.ndarray,
        weights: np.ndarray,
    ) -> dict[Phase, tuple[np.ndarray, np.ndarray]]:
        """Randomly split matrices into train / validation / test."""
        n = len(matrices)
        idx = np.random.permutation(n)

        n_train = int(n * SPLIT_RATIOS[Phase.TRAIN])
        n_val   = int(n * SPLIT_RATIOS[Phase.VALIDATION])

        idx_train = idx[:n_train]
        idx_val   = idx[n_train: n_train + n_val]
        idx_test  = idx[n_train + n_val:]

        return {
            Phase.TRAIN:      (matrices[idx_train], weights[idx_train]),
            Phase.VALIDATION: (matrices[idx_val],   weights[idx_val]),
            Phase.TEST:       (matrices[idx_test],  weights[idx_test]),
        }

    def _save(
        self,
        matrices: np.ndarray,
        weights: np.ndarray,
        num_criteria: int,
        phase: Phase,
        stratum_name: str,
        sigma: float,
        alpha: float,
    ) -> None:
        """Save matrices and weights as .npy files.

        File naming convention
        ----------------------
        matrices_sigma{sigma:.2f}.npy  – encodes the log-normal noise σ used
        weights_alpha{alpha:.2f}.npy   – encodes the Dirichlet α used
        (dots in the float are replaced with underscores, e.g. sigma0_90, alpha2_00)
        """
        save_dir = os.path.join(
            self.base_output_dir,
            f"criteria{num_criteria}",
            phase.value,
            stratum_name,
        )
        os.makedirs(save_dir, exist_ok=True)

        matrices_fname = f"matrices_sigma{sigma:.2f}.npy".replace(".", "_").replace("_npy", ".npy")
        weights_fname  = f"weights_alpha{alpha:.2f}.npy".replace(".", "_").replace("_npy", ".npy")

        np.save(os.path.join(save_dir, matrices_fname), matrices)
        np.save(os.path.join(save_dir, weights_fname),  weights)

        logger.info(
            "Saved %5d matrices → %s  [%s | %s]",
            len(matrices), save_dir, matrices_fname, weights_fname,
        )
