"""Lightweight stale check for benchmark-paper generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from polymarket_research.scripts.common import resolve_repo_root


STALE_RELEASE_PATTERNS = [
    ("old_admitted_markets", "4,812"),
    ("old_terminal_examples", "8,436"),
    ("old_decisiveness_examples", "2,754"),
    ("old_repricing_examples", "209,884"),
    ("old_terminal_yes_rate", "54.2\\%"),
    ("old_short_convergence_rate", "18.7\\%"),
    ("old_repricing_event_rate", "7.3\\%"),
    ("old_validation_split_phrase", "Train / validation / test"),
    ("old_validation_split_prose", "validation examples fall between"),
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_hashes(records: list[dict[str, Any]], *, label: str) -> list[str]:
    warnings: list[str] = []
    for record in records:
        path = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", "")).strip()
        if not path.exists():
            warnings.append(f"{label}: missing file for {record.get('role', 'unknown')}: {path}")
            continue
        actual = _sha256_file(path)
        if expected and actual != expected:
            warnings.append(
                f"{label}: hash mismatch for {record.get('role', 'unknown')}: {path} "
                f"(expected {expected}, got {actual})"
            )
    return warnings


def check_paper_artifacts(
    *,
    generated_dir: Path,
    paper_path: Path,
) -> tuple[list[str], list[str]]:
    manifest_path = generated_dir / "paper_manifest.json"
    warnings: list[str] = []
    notes: list[str] = []

    if not manifest_path.exists():
        return [f"missing generated manifest: {manifest_path}"], notes

    manifest = _read_json(manifest_path)
    warnings.extend(_check_hashes(manifest.get("input_files", []), label="input"))
    warnings.extend(_check_hashes(manifest.get("output_files", []), label="output"))

    if not paper_path.exists():
        warnings.append(f"missing paper: {paper_path}")
    else:
        paper_text = paper_path.read_text(encoding="utf-8")
        for name, pattern in STALE_RELEASE_PATTERNS:
            if pattern in paper_text:
                warnings.append(f"stale paper pattern still present ({name}): {pattern}")

    missing = manifest.get("missing_experiment_artifacts", [])
    if missing:
        notes.append("missing experiment artifacts reported by generated manifest:")
        for item in missing:
            notes.append(f"- {item.get('id', 'unknown')}: {item.get('description', '')}")

    return warnings, notes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to auto-detection.")
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Generated paper artifact directory. Defaults to <repo>/writing/benchmark/generated.",
    )
    parser.add_argument(
        "--paper-path",
        default=None,
        help="Paper path. Defaults to <repo>/writing/benchmark/benchmark_paper_v11.tex.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when warnings are present.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    generated_dir = Path(args.generated_dir) if args.generated_dir else repo_root / "writing" / "benchmark" / "generated"
    if not generated_dir.is_absolute():
        generated_dir = repo_root / generated_dir
    paper_path = Path(args.paper_path) if args.paper_path else repo_root / "writing" / "benchmark" / "benchmark_paper_v11.tex"
    if not paper_path.is_absolute():
        paper_path = repo_root / paper_path

    warnings, notes = check_paper_artifacts(generated_dir=generated_dir, paper_path=paper_path)
    if warnings:
        print("[paper stale-check] warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("[paper stale-check] no stale release-count or hash warnings")

    for note in notes:
        print(f"[paper stale-check] {note}")

    if args.strict and warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
