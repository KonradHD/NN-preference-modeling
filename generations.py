"""
Monte Carlo AHP dataset generation script.

Generates pairwise comparison matrices for all criteria sizes (n=3..9),
stratified by Consistency Ratio:
  - High consistency   (CR <= 0.10) -> 40 %
  - Moderate           (0.10 < CR <= 0.33) -> 50 %
  - Extreme            (CR > 0.33) -> 10 %

Each stratum is split into train / validation / test (80 / 10 / 10 %).

Usage (from project root):
  python generations.py                                    # brakujace dane
  python generations.py --force                            # wszystko od nowa
  python generations.py --criteria 4 5                     # wybrane rozmiary
  python generations.py --alpha 3.0                        # Dirichlet alfa=3.0
  python generations.py --sigma-low 3:0.85 4:0.75          # nadpisz sigma_low dla n=3 i n=4
  python generations.py --sigma-high 3:1.2 5:0.95          # nadpisz sigma_high dla n=3 i n=5
  python generations.py --sigma-low 0.7 --sigma-high 1.0   # jeden wspolny override dla wszystkich n
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Per-criteria target matrix counts (total ~490 000 across n=3..9)
# ---------------------------------------------------------------------------
MATRICES_PER_CRITERIA: dict[int, int] = {
    3:  40_000,
    4:  50_000,
    5:  60_000,
    6:  70_000,
    7:  80_000,
    8:  90_000,
    9: 100_000,
}

OUTPUT_DIR = os.path.join("data", "monte-carlo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _criteria_is_complete(n: int) -> bool:
    """Return True if all 18 expected .npy files already exist for criteria n."""
    strata = ["low_inconsistency", "moderate_inconsistency", "high_inconsistency"]
    phases = ["train", "validation", "test"]
    for phase in phases:
        for stratum in strata:
            d = os.path.join(OUTPUT_DIR, f"criteria{n}", phase, stratum)
            # Accept any matrices_*.npy / weights_*.npy (new naming) OR legacy names
            has_matrices = (
                bool([f for f in os.listdir(d) if f.startswith("matrices") and f.endswith(".npy")])
                if os.path.isdir(d) else False
            )
            has_weights = (
                bool([f for f in os.listdir(d) if f.startswith("weights") and f.endswith(".npy")])
                if os.path.isdir(d) else False
            )
            if not has_matrices or not has_weights:
                return False
    return True


def _parse_sigma_overrides(values: list[str] | None) -> dict[int, float] | float | None:
    """Parse sigma override argument.

    Accepted formats:
      - Single float,  e.g. "0.75"  → applies to all criteria (returned as float)
      - N:VALUE pairs, e.g. "3:0.9" "4:0.8"  → per-criteria dict

    Returns a dict, a float, or None if values is None/empty.
    """
    if not values:
        return None
    if len(values) == 1 and ":" not in values[0]:
        # Single global value
        try:
            return float(values[0])
        except ValueError:
            print(f"Blad: nieprawidlowa wartosc sigma '{values[0]}'", file=sys.stderr)
            sys.exit(1)
    # N:VALUE pairs
    result: dict[int, float] = {}
    for token in values:
        if ":" not in token:
            print(
                f"Blad: oczekiwano formatu N:WARTOSC (np. '3:0.9'), otrzymano '{token}'",
                file=sys.stderr,
            )
            sys.exit(1)
        n_str, v_str = token.split(":", 1)
        try:
            result[int(n_str)] = float(v_str)
        except ValueError:
            print(f"Blad: nieprawidlowy wpis '{token}'", file=sys.stderr)
            sys.exit(1)
    return result


def _resolve_sigma(overrides: dict[int, float] | float | None, n: int) -> float | None:
    """Return the sigma to use for a given n, or None to use the default."""
    if overrides is None:
        return None
    if isinstance(overrides, float):
        return overrides
    return overrides.get(n, None)  # None = use default for this n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generuje pelny dataset Monte Carlo AHP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przyklady:\n"
            "  python generations.py\n"
            "  python generations.py --criteria 4 5\n"
            "  python generations.py --force\n"
            "  python generations.py --alpha 3.0\n"
            "  python generations.py --sigma-low 0.7           # wspolna wartosc dla wszystkich n\n"
            "  python generations.py --sigma-low 3:0.85 4:0.75 # per-criteria\n"
            "  python generations.py --sigma-high 3:1.2 5:0.95\n"
        ),
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        type=int,
        default=list(range(3, 10)),
        metavar="N",
        help="Rozmiary macierzy do wygenerowania (3-9). Domyslnie: wszystkie.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regeneruj nawet jesli pliki juz istnieja.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        metavar="ALPHA",
        help=(
            "Parametr koncentracji Dirichleta alfa (skalar, rozglosniony na "
            "wszystkie wymiary). Domyslnie: 2.0."
        ),
    )
    parser.add_argument(
        "--sigma-low",
        nargs="+",
        metavar="[N:]VALUE",
        help=(
            "Override sigma_low (szum dla high+moderate strata). "
            "Podaj jeden float (globalny) lub pary N:VALUE (per-criteria), "
            "np. --sigma-low 0.7  lub  --sigma-low 3:0.85 4:0.75"
        ),
    )
    parser.add_argument(
        "--sigma-high",
        nargs="+",
        metavar="[N:]VALUE",
        help=(
            "Override sigma_high (szum dla extreme stratum). "
            "Podaj jeden float (globalny) lub pary N:VALUE (per-criteria), "
            "np. --sigma-high 1.1  lub  --sigma-high 3:1.3 4:1.1"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50_000,
        metavar="N",
        help="Liczba macierzy w jednej partii. Domyslnie: 50 000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Globalny seed NumPy. Uzyj -1 aby wylaczyc. Domyslnie: 42.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)
    args = _parse_args()

    # Walidacja criteria
    invalid = [n for n in args.criteria if not (3 <= n <= 9)]
    if invalid:
        logger.error("Nieprawidlowe rozmiary (musi byc 3-9): %s", invalid)
        sys.exit(1)

    sigma_low_overrides  = _parse_sigma_overrides(args.sigma_low)
    sigma_high_overrides = _parse_sigma_overrides(args.sigma_high)

    # Import po konfiguracji logowania
    from data_augmentation.monte_carlo_data_generator import MonteCarloDataGenerator

    seed = args.seed if args.seed >= 0 else None

    generator = MonteCarloDataGenerator(
        base_output_dir=OUTPUT_DIR,
        alpha=args.alpha,
        batch_size=args.batch,
        seed=seed,
    )

    criteria_to_run = sorted(args.criteria)

    if not args.force:
        skipped = [n for n in criteria_to_run if _criteria_is_complete(n)]
        criteria_to_run = [n for n in criteria_to_run if not _criteria_is_complete(n)]
        if skipped:
            logger.info(
                "Pomijam juz kompletne: %s  (--force zeby regenerowac)", skipped
            )

    if not criteria_to_run:
        logger.info("Nic do wygenerowania – wszystkie kryteria sa juz kompletne.")
        return

    total_start = time.time()
    grand_total = 0

    for n in criteria_to_run:
        n_matrices = MATRICES_PER_CRITERIA[n]
        sl = _resolve_sigma(sigma_low_overrides, n)
        sh = _resolve_sigma(sigma_high_overrides, n)

        separator = "=" * 60
        logger.info(separator)
        logger.info(
            "  criteria=%d | total=%d | alpha=%.2f | sigma_low=%s | sigma_high=%s",
            n, n_matrices, args.alpha,
            f"{sl:.2f}" if sl is not None else "default",
            f"{sh:.2f}" if sh is not None else "default",
        )
        logger.info(separator)

        generator.generate(
            num_criteria=n,
            total_matrices=n_matrices,
            sigma_low=sl,
            sigma_high=sh,
        )
        grand_total += n_matrices

    elapsed = time.time() - total_start
    logger.info(
        "Gotowe. Wygenerowano ~%d macierzy dla %s w %.1f s.",
        grand_total, criteria_to_run, elapsed,
    )


if __name__ == "__main__":
    main()
