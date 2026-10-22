"""
Entry-point script for generating the full Monte Carlo AHP dataset.

Run from the project root:
  python generate_monte_carlo.py

Optional flags:
  --criteria    3 4 5 6 7 8 9           (default: all 3-9)
  --total       N_per_criteria           (default: see MATRICES_PER_CRITERIA)
  --output      data/monte-carlo         (default)
  --alpha       2.0                      (Dirichlet concentration, default 2.0)
  --sigma-low   [N:]VALUE ...            (override low-noise sigma, per-criteria or global)
  --sigma-high  [N:]VALUE ...            (override high-noise sigma, per-criteria or global)
  --batch       50000                    (batch size, default 50 000)
  --seed        42                       (random seed, default 42)
  --force                                (regenerate even if files already exist)

Dataset specification (from thesis):
  - n in {3, ..., 9}
  - Dirichlet(alpha) weights -> perfectly consistent matrix -> multiplicative
    log-normal noise -> Saaty-scale discretization
  - Stratified by CR:
      . High consistency   CR <= 0.10 -> 40 %
      . Moderate           0.10 < CR <= 0.33 -> 50 %
      . Extreme            CR > 0.33 -> 10 %
  - Train / Validation / Test split: 80 / 10 / 10 %
  - Scale: 250 000 - 750 000 total matrices (split across n sizes)

Sigma override examples:
  --sigma-low 0.7                  # global override for all criteria
  --sigma-low 3:0.85 4:0.75        # per-criteria (N:VALUE pairs)
  --sigma-high 3:1.2 5:0.95        # per-criteria for sigma_high
"""

from __future__ import annotations

import argparse
import logging
import time
import sys
import os


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# Number of matrices to generate per criteria size.
# Follows the thesis recommendation:
#   - Smaller n -> fewer needed (lower input dimensionality)
#   - Larger n -> exponentially more needed for good generalization
# Total across all 7 sizes ~= 490 000
MATRICES_PER_CRITERIA: dict[int, int] = {
    3:  40,
    4:  50,
    5:  60,
    6:  70,
    7:  80,
    8:  90,
    9: 100,
}


# ---------------------------------------------------------------------------
# Sigma override helpers
# ---------------------------------------------------------------------------

def _parse_sigma_overrides(values: list[str] | None) -> dict[int, float] | float | None:
    """Parse --sigma-low / --sigma-high argument.

    Accepted formats:
      - Single float, e.g. "0.75"    -> applies to all criteria (returned as float)
      - N:VALUE pairs, e.g. "3:0.9"  -> per-criteria dict
    """
    if not values:
        return None
    if len(values) == 1 and ":" not in values[0]:
        try:
            return float(values[0])
        except ValueError:
            print(f"Error: invalid sigma value '{values[0]}'", file=sys.stderr)
            sys.exit(1)
    result: dict[int, float] = {}
    for token in values:
        if ":" not in token:
            print(
                f"Error: expected N:VALUE format (e.g. '3:0.9'), got '{token}'",
                file=sys.stderr,
            )
            sys.exit(1)
        n_str, v_str = token.split(":", 1)
        try:
            result[int(n_str)] = float(v_str)
        except ValueError:
            print(f"Error: invalid entry '{token}'", file=sys.stderr)
            sys.exit(1)
    return result


def _resolve_sigma(overrides: dict[int, float] | float | None, n: int) -> float | None:
    """Return the sigma override for criteria n, or None to use the default."""
    if overrides is None:
        return None
    if isinstance(overrides, float):
        return overrides
    return overrides.get(n, None)


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _criteria_is_complete(output_dir: str, n: int) -> bool:
    """Return True if all 18 .npy files already exist for criteria n."""
    strata = ["high_consistency", "moderate_consistency", "extreme_consistency"]
    phases = ["train", "validation", "test"]
    for phase in phases:
        for stratum in strata:
            d = os.path.join(output_dir, f"criteria{n}", phase, stratum)
            if not os.path.isdir(d):
                return False
            has_matrices = any(
                f.startswith("matrices") and f.endswith(".npy")
                for f in os.listdir(d)
            )
            has_weights = any(
                f.startswith("weights") and f.endswith(".npy")
                for f in os.listdir(d)
            )
            if not has_matrices or not has_weights:
                return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo AHP dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_monte_carlo.py\n"
            "  python generate_monte_carlo.py --criteria 4 5 --force\n"
            "  python generate_monte_carlo.py --alpha 3.0\n"
            "  python generate_monte_carlo.py --sigma-low 0.7 --sigma-high 1.1\n"
            "  python generate_monte_carlo.py --sigma-low 3:0.85 4:0.75\n"
        ),
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        type=int,
        default=list(range(3, 10)),
        metavar="N",
        help="Which criteria sizes to generate (space-separated integers 3-9). Default: all.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override total matrices per criteria size. "
            "If not set, uses the per-size defaults in MATRICES_PER_CRITERIA."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "monte-carlo"),
        metavar="DIR",
        help="Root output directory. Default: data/monte-carlo.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        metavar="ALPHA",
        help="Dirichlet concentration parameter alpha (scalar, broadcast to all dimensions). Default: 2.0.",
    )
    parser.add_argument(
        "--sigma-low",
        nargs="+",
        metavar="[N:]VALUE",
        help=(
            "Override sigma_low (noise for high+moderate strata). "
            "Provide a single float (global) or N:VALUE pairs (per-criteria), "
            "e.g. --sigma-low 0.7  or  --sigma-low 3:0.85 4:0.75"
        ),
    )
    parser.add_argument(
        "--sigma-high",
        nargs="+",
        metavar="[N:]VALUE",
        help=(
            "Override sigma_high (noise for extreme stratum). "
            "Provide a single float (global) or N:VALUE pairs (per-criteria), "
            "e.g. --sigma-high 1.1  or  --sigma-high 3:1.3 4:1.1"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50_000,
        metavar="N",
        help="Number of matrices per generation batch. Default: 50 000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Global random seed (use -1 to disable). Default: 42.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate even if output files already exist.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    args = parse_args()

    # Import here so logging is already configured
    from data_augmentation.monte_carlo_data_generator import MonteCarloDataGenerator

    logger = logging.getLogger(__name__)

    invalid = [n for n in args.criteria if not (3 <= n <= 9)]
    if invalid:
        logger.error("Invalid criteria sizes (must be 3-9): %s", invalid)
        sys.exit(1)

    sigma_low_overrides  = _parse_sigma_overrides(args.sigma_low)
    sigma_high_overrides = _parse_sigma_overrides(args.sigma_high)

    seed = args.seed if args.seed >= 0 else None

    generator = MonteCarloDataGenerator(
        base_output_dir=args.output,
        alpha=args.alpha,
        batch_size=args.batch,
        seed=seed,
    )

    criteria_to_run = sorted(args.criteria)

    if not args.force:
        skipped = [n for n in criteria_to_run if _criteria_is_complete(args.output, n)]
        criteria_to_run = [n for n in criteria_to_run if not _criteria_is_complete(args.output, n)]
        if skipped:
            logger.info(
                "Skipping already-complete criteria: %s  (use --force to regenerate)",
                skipped,
            )

    if not criteria_to_run:
        logger.info("Nothing to generate – all requested criteria are already complete.")
        return

    total_start = time.time()
    grand_total = 0

    for n in criteria_to_run:
        n_matrices = args.total if args.total is not None else MATRICES_PER_CRITERIA[n]
        sl = _resolve_sigma(sigma_low_overrides, n)
        sh = _resolve_sigma(sigma_high_overrides, n)

        logger.info("=" * 57)
        logger.info(
            "  criteria = %d  |  target = %d  |  alpha = %.2f  |  "
            "sigma_low = %s  |  sigma_high = %s",
            n, n_matrices, args.alpha,
            f"{sl:.2f}" if sl is not None else "default",
            f"{sh:.2f}" if sh is not None else "default",
        )
        logger.info("=" * 57)

        generator.generate(
            num_criteria=n,
            total_matrices=n_matrices,
            sigma_low=sl,
            sigma_high=sh,
        )
        grand_total += n_matrices

    elapsed = time.time() - total_start
    logger.info(
        "All done. Generated ~%d matrices across %s criteria in %.1fs.",
        grand_total,
        criteria_to_run,
        elapsed,
    )


if __name__ == "__main__":
    main()
