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

from analytic_methods.EVM_method import EVM
from data_augmentation.generator.monte_carlo_generator import MonteCarloGenerator
from utils.phase import Phase
from data_augmentation.monte_carlo_paramerets import CR_STRATA, SPLIT_RATIOS, _SIGMA_LOW, _SIGMA_HIGH

logger = logging.getLogger(__name__)


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
        total_matrices: int | None = None,
        sigma_low: float | None = None,
        sigma_high: float | None = None,
        needs: dict[str, int] | None = None,
    ) -> None:
        if needs is not None:
            needs = dict(needs)
            for stratum_name in CR_STRATA:
                if stratum_name not in needs:
                    raise ValueError(f"Missing required stratum key '{stratum_name}' in needs")
            if total_matrices is None:
                total_matrices = sum(needs.values())
        else:
            if total_matrices is None:
                raise ValueError("Either total_matrices or needs must be provided.")
            needs = {
                name: int(total_matrices * frac)
                for name, (_, _, frac) in CR_STRATA.items()
            }
            diff = total_matrices - sum(needs.values())
            needs["moderate_consistency"] += diff

        logger.info(
            "Generating %d matrices for criteria=%d with needs=%s",
            total_matrices,
            num_criteria,
            needs,
        )
        t0 = time.time()

        sigma_low = sigma_low if sigma_low is not None else _SIGMA_LOW.get(num_criteria, 0.70)
        sigma_high = sigma_high if sigma_high is not None else _SIGMA_HIGH.get(num_criteria, 1.00)
        logger.info("  sigma_low=%.2f  sigma_high=%.2f", sigma_low, sigma_high)

        pools: dict[str, dict[str, list[np.ndarray] | int]] = {
            name: {"matrices": [], "weights": [], "count": 0}
            for name in CR_STRATA
        }

        low_target = needs["high_consistency"] + needs["moderate_consistency"]
        budget_low = max(int(low_target * 3), 100)
        generated_low = 0
        while (
            pools["high_consistency"]["count"] < needs["high_consistency"]
            or pools["moderate_consistency"]["count"] < needs["moderate_consistency"]
        ) and generated_low < budget_low:
            needed_low = (
                (needs["high_consistency"] - pools["high_consistency"]["count"])
                + (needs["moderate_consistency"] - pools["moderate_consistency"]["count"])
            )
            batch = min(self.batch_size, max(50, needed_low * 2))
            self._generate_and_route(num_criteria, sigma_low, batch, pools, needs)
            generated_low += batch

        extreme_target = needs["extreme_consistency"]
        budget_extreme = max(int(extreme_target * 5), 100)
        generated_extreme = 0
        while (
            pools["extreme_consistency"]["count"] < needs["extreme_consistency"]
            and generated_extreme < budget_extreme
        ):
            needed_extreme = needs["extreme_consistency"] - pools["extreme_consistency"]["count"]
            batch = min(self.batch_size, max(50, needed_extreme * 3))
            self._generate_and_route(num_criteria, sigma_high, batch, pools, needs)
            generated_extreme += batch

        adaptive_sigma_low = sigma_low
        adaptive_sigma_extreme = sigma_high
        adaptive_sigma_moderate = (sigma_low + sigma_high) / 2.0
        max_attempts = 150
        attempt = 0

        while any(pools[name]["count"] < needs[name] for name in CR_STRATA):
            attempt += 1
            if attempt > max_attempts:
                missing_info = {
                    name: f"{pools[name]['count']}/{needs[name]}"
                    for name in CR_STRATA
                }
                raise RuntimeError(
                    f"Failed to fill all strata after {max_attempts} attempts. Status: {missing_info}"
                )

            if pools["extreme_consistency"]["count"] < needs["extreme_consistency"]:
                missing = needs["extreme_consistency"] - pools["extreme_consistency"]["count"]
                adaptive_sigma_extreme = max(adaptive_sigma_extreme * 1.20, adaptive_sigma_extreme + 0.15)
                batch = min(self.batch_size, max(50, missing * 3))
                logger.info(
                    "Adaptive fallback for extreme_consistency: missing %d, increased sigma to %.2f",
                    missing,
                    adaptive_sigma_extreme,
                )
                self._generate_and_route(num_criteria, adaptive_sigma_extreme, batch, pools, needs)

            if pools["high_consistency"]["count"] < needs["high_consistency"]:
                missing = needs["high_consistency"] - pools["high_consistency"]["count"]
                adaptive_sigma_low = max(0.05, adaptive_sigma_low * 0.75)
                batch = min(self.batch_size, max(50, missing * 3))
                logger.info(
                    "Adaptive fallback for high_consistency: missing %d, decreased sigma to %.2f",
                    missing,
                    adaptive_sigma_low,
                )
                self._generate_and_route(num_criteria, adaptive_sigma_low, batch, pools, needs)

            if pools["moderate_consistency"]["count"] < needs["moderate_consistency"]:
                missing = needs["moderate_consistency"] - pools["moderate_consistency"]["count"]
                adaptive_sigma_moderate = 0.70
                batch = min(self.batch_size, max(50, missing * 3))
                logger.info(
                    "Adaptive fallback for moderate_consistency: missing %d, using sigma=%.2f",
                    missing,
                    adaptive_sigma_moderate,
                )
                self._generate_and_route(num_criteria, adaptive_sigma_moderate, batch, pools, needs)

        logger.info(
            "Pools filled in %.1fs | high=%d mod=%d ext=%d",
            time.time() - t0,
            pools["high_consistency"]["count"],
            pools["moderate_consistency"]["count"],
            pools["extreme_consistency"]["count"],
        )

        stratum_sigma: dict[str, float] = {
            "high_consistency": sigma_low,
            "moderate_consistency": sigma_low,
            "extreme_consistency": sigma_high,
        }

        for stratum_name in CR_STRATA:
            if needs[stratum_name] == 0:
                continue
            matrices = np.concatenate(pools[stratum_name]["matrices"], axis=0)[: needs[stratum_name]]
            weights = np.concatenate(pools[stratum_name]["weights"], axis=0)[: needs[stratum_name]]

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

    def _route_batch(
        self,
        pools: dict[str, dict[str, list[np.ndarray] | int]],
        mats: np.ndarray,
        wts: np.ndarray,
        crs: np.ndarray,
        needs: dict[str, int],
    ) -> int:
        added_total = 0
        for name, (cr_min, cr_max, _) in CR_STRATA.items():
            missing = needs[name] - pools[name]["count"]
            if missing <= 0:
                continue
            if cr_max == float("inf"):
                mask = crs > cr_min
            else:
                mask = (crs > cr_min) & (crs <= cr_max)

            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            take = min(len(idx), missing)
            sel = idx[:take]
            pools[name]["matrices"].append(mats[sel])
            pools[name]["weights"].append(wts[sel])
            pools[name]["count"] += take
            added_total += take
        return added_total

    def _generate_and_route(
        self,
        num_criteria: int,
        sigma: float,
        batch_size: int,
        pools: dict[str, dict[str, list[np.ndarray] | int]],
        needs: dict[str, int],
    ) -> int:
        gen = MonteCarloGenerator(
            num_criteria=num_criteria,
            alpha=self.alpha,
            noise_sigma=sigma,
        )
        mats, wts = gen.generate(batch_size)
        crs = self._evm.compute_batch(mats)[1]
        return self._route_batch(pools, mats, wts, crs, needs)


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
            "Saved %5d matrices -> %s  [%s | %s]",
            len(matrices), save_dir, matrices_fname, weights_fname,
        )

