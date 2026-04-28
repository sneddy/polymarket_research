"""CLI for materializing a canonical dataset from internal raw sources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from polymarket_research.data.canonical import CanonicalDataset, CanonicalDatasetBuilder
from polymarket_research.data.raw import RawExternalCovariates, RawMarketHandle
from polymarket_research.scripts.common import (
    VALID_SOURCES,
    default_canonical_cache_dir,
    default_external_covariates_path,
    default_raw_snapshot_dir,
    log_message,
    log_stage,
    print_frame,
    resolve_repo_root,
)
from polymarket_research.utils.data import default_db_path


@dataclass(frozen=True)
class CanonicalBuildResult:
    """Structured result returned by the canonical build script helpers."""

    canonical: CanonicalDataset
    canonical_dir: Path
    db_path: Path
    raw_snapshot_dir: Path | None = None


def build_canonical_dataset(
    *,
    repo_root: str | Path | None = None,
    source: str = "polymarket",
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    market_limit: int | None = None,
    market_order: str | None = None,
    include_raw_trades: bool = False,
    include_external_covariates: bool = True,
    external_covariates_path: str | Path | None = None,
    resolved_only: bool = True,
    save_raw_snapshot: bool = False,
    raw_snapshot_dir: str | Path | None = None,
    show_progress: bool = True,
) -> CanonicalBuildResult:
    """Load raw export tables, build a canonical dataset, and persist it."""
    prefix = "[build_canonical]"
    root = resolve_repo_root(repo_root)
    normalized_source = str(source)
    canonical_dir = Path(output_dir) if output_dir is not None else default_canonical_cache_dir(root, normalized_source)
    snapshot_dir = None
    if save_raw_snapshot:
        snapshot_dir = Path(raw_snapshot_dir) if raw_snapshot_dir is not None else default_raw_snapshot_dir(root, normalized_source)
    resolved_db_path = Path(db_path) if db_path is not None else default_db_path(root, source=normalized_source)

    log_message(
        prefix,
        f"config: source={normalized_source} db_path={resolved_db_path} canonical_dir={canonical_dir}",
    )
    if snapshot_dir is not None:
        log_message(prefix, f"raw_snapshot_dir={snapshot_dir}")

    handle = RawMarketHandle(
        source=normalized_source,
        db_path=resolved_db_path,
        cache_dir=snapshot_dir,
    )
    with log_stage(prefix, "load raw bundle"):
        raw_bundle = handle.load_bundle(
            include_market_universe=False,
            include_download_manifest=True,
            include_probabilities=True,
            include_raw_trades=bool(include_raw_trades),
            market_limit=market_limit,
            market_order=market_order,
            show_progress=show_progress,
        )
    print_frame(f"{prefix} raw summary", raw_bundle.summary())
    if snapshot_dir is not None:
        with log_stage(prefix, "save raw snapshot"):
            handle.snapshot().save_bundle(raw_bundle)

    raw_external = None
    if include_external_covariates:
        covariates_root = (
            Path(external_covariates_path)
            if external_covariates_path is not None
            else default_external_covariates_path(root)
        )
        if covariates_root.exists():
            with log_stage(prefix, "prepare external covariates"):
                raw_external = RawExternalCovariates(path=covariates_root)
        else:
            log_message(prefix, f"external covariates path not found, skipping: {covariates_root}")

    with log_stage(prefix, "build canonical dataset"):
        canonical = CanonicalDatasetBuilder(
            raw_dataset=raw_bundle,
            raw_external=raw_external,
            resolved_only=bool(resolved_only),
            show_progress=show_progress,
        ).build()
    with log_stage(prefix, "save canonical dataset"):
        canonical.save(canonical_dir)
    return CanonicalBuildResult(
        canonical=canonical,
        canonical_dir=canonical_dir,
        db_path=resolved_db_path,
        raw_snapshot_dir=snapshot_dir,
    )


def build_parser(*, include_help: bool = True) -> argparse.ArgumentParser:
    """Create the CLI parser for canonical materialization."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=include_help)
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root. Defaults to discovery from cwd.")
    parser.add_argument("--source", choices=VALID_SOURCES, default="polymarket")
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite path. Defaults to the source-specific db in repo_root/db.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Canonical output directory. Defaults to the internal canonical cache path.")
    parser.add_argument("--market-limit", type=int, default=None)
    parser.add_argument("--market-order", choices=("latest", "largest"), default=None)
    parser.add_argument("--include-raw-trades", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-external-covariates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-covariates-path", type=Path, default=None)
    parser.add_argument("--resolved-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw-snapshot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--raw-snapshot-dir", type=Path, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the canonical build CLI."""
    args = build_parser().parse_args(argv)
    result = build_canonical_dataset(
        repo_root=args.repo_root,
        source=args.source,
        db_path=args.db_path,
        output_dir=args.output_dir,
        market_limit=args.market_limit,
        market_order=args.market_order,
        include_raw_trades=args.include_raw_trades,
        include_external_covariates=args.include_external_covariates,
        external_covariates_path=args.external_covariates_path,
        resolved_only=args.resolved_only,
        save_raw_snapshot=args.save_raw_snapshot,
        raw_snapshot_dir=args.raw_snapshot_dir,
        show_progress=args.show_progress,
    )
    print(f"[build_canonical] wrote canonical dataset to: {result.canonical_dir}")
    if result.raw_snapshot_dir is not None:
        print(f"[build_canonical] wrote raw snapshot to: {result.raw_snapshot_dir}")
    print_frame("[build_canonical] canonical summary", result.canonical.summary())
    print_frame("[build_canonical] canonical status", result.canonical.status())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
