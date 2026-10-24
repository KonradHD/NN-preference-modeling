from utils.phase import Phase
import numpy as np

# Number of matrices to generate per criteria size.
# Follows the thesis recommendation:
#   - Smaller n -> fewer needed (lower input dimensionality)
#   - Larger n -> exponentially more needed for good generalization
# Total across all 7 sizes ~= 490 000
MATRICES_PER_CRITERIA: dict[int, int] = {
    3: 250_000,
    4: 350_000,
    5: 450_000,
    6: 550_000,
    7: 650_000,
    8: 750_000,
    9: 850_000,
}

CR_STRATA: dict[str, tuple[float, float, float]] = {
    # name: (cr_min_exclusive, cr_max_inclusive, fraction_of_total)
    "low_inconsistency":     (0.00, 0.10, 0.40),
    "moderate_inconsistency": (0.10, 0.33, 0.50),
    "high_inconsistency":  (0.33, float("inf"), 0.10),
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

# Saaty's fundamental scale: 1, 2, ..., 9 and their reciprocals
SAATY_SCALE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
# All possible values: reciprocals + scale
SAATY_VALUES = np.concatenate([1.0 / SAATY_SCALE[::-1][:-1], SAATY_SCALE])  # 1/9, 1/8, ..., 1, 2, ..., 9


