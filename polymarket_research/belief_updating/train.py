"""
MVP outcome experiment for contextual belief updating.

Runs the four-rung evaluation protocol from the paper:

    Rung 1  stale-only            x_A^stale → y_A
    Rung 2  stale + raw context   x_A^stale + C_t^raw → y_A     (aggregated mean/max)
    Rung 3  stale + embedding     x_A^stale + z_t → y_A          (DeepSets encoder)
    Rung 4  stale + corrupted     x_A^stale + z̃_t → y_A          (shuffled context control)

Rungs 1 and 2 use sklearn HistGradientBoostingClassifier (same as existing benchmarks).
Rungs 3 and 4 use the PyTorch DeepSetsEncoder + BeliefUpdatingPredictor.

The key result the paper needs: Perf(rung 3) ≈ Perf(rung 2) >> Perf(rung 1),
and Perf(rung 4) ≈ Perf(rung 1) (shuffled context does not help → the gain
in rung 3 comes from genuine contextual information, not extra parameters).

All rungs use a strict grouped out-of-time train/test split on end_date.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from polymarket_research.belief_updating.dataset import (
    AGG_CONTEXT_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    STALE_FEATURE_NAMES,
    BeliefUpdatingManifest,
)
from polymarket_research.belief_updating.model import (
    BeliefUpdatingPredictor,
    BeliefUpdatingTorchDataset,
    DeepSetsEncoder,
    shuffle_context_across_batch,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BeliefUpdatingMVPConfig:
    """
    Training configuration for the MVP outcome experiment.

    Parameters
    ----------
    test_fraction:
        Fraction of examples (by end_date order) held out for testing.
    n_epochs:
        Training epochs for PyTorch rungs (3 and 4).
    batch_size:
        Mini-batch size for PyTorch rungs.
    learning_rate:
        Adam learning rate for PyTorch rungs.
    encoder_hidden_dim:
        Hidden width of the DeepSets per-element MLP.
    encoder_output_dim:
        Embedding dimension d_z.
    max_context_size:
        Maximum siblings per example (context is padded/truncated to this).
    device:
        PyTorch device string.  "cpu" by default; set "cuda" or "mps" if available.
    random_state:
        Seed for reproducibility in sklearn and torch.
    gbm_max_iter:
        Max boosting rounds for HistGBT (rungs 1 and 2).
    log_every_epochs:
        How often to print epoch-level progress for PyTorch rungs.
    show_epoch_progress:
        Whether to print epoch-level loss traces for PyTorch rungs.
    """

    test_fraction: float = 0.25
    n_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    encoder_hidden_dim: int = 64
    encoder_output_dim: int = 32
    max_context_size: int = 16
    device: str = "cpu"
    random_state: int = 42
    gbm_max_iter: int = 300
    log_every_epochs: int = 1
    show_epoch_progress: bool = True


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class BeliefUpdatingMVPRungMetrics:
    """Metrics for one rung evaluated on the MVP test split."""

    rung: str
    n_train: int
    n_test: int
    log_loss: float
    brier: float
    roc_auc: float
    train_time_s: float
    notes: str = ""
    training_history: list[dict[str, float]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "log_loss": round(self.log_loss, 5),
            "brier": round(self.brier, 5),
            "roc_auc": round(self.roc_auc, 5),
            "train_time_s": round(self.train_time_s, 1),
            "notes": self.notes,
        }


@dataclass
class BeliefUpdatingMVPResults:
    """
    Collected metrics across all four rungs.

    Attributes
    ----------
    rungs:
        List of BeliefUpdatingMVPRungMetrics in rung order.
    config:
        The BeliefUpdatingMVPConfig used for this run.

    Key figures from the paper's hypothesis:
      - ``gain_raw_over_stale``: rung2.log_loss − rung1.log_loss (should be negative)
      - ``gain_embedding_over_stale``: rung3.log_loss − rung1.log_loss (should be similar)
      - ``compression_retention``: how much of the raw-context gain is retained by the
        compact embedding
    """

    rungs: list[BeliefUpdatingMVPRungMetrics]
    config: BeliefUpdatingMVPConfig

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.rungs])

    def training_histories(self) -> dict[str, pd.DataFrame]:
        """Return per-rung training traces when available."""
        return {
            rung.rung: pd.DataFrame(rung.training_history)
            for rung in self.rungs
            if rung.training_history
        }

    def compression_summary(self) -> pd.DataFrame:
        """
        Report the paper's central compression claim:

            gain_raw    = Δ log-loss from stale-only → stale+raw
            gain_embed  = Δ log-loss from stale-only → stale+embed
            retention   = gain_embed / gain_raw  (ideally close to 1)
        """
        metrics = {r.rung: r for r in self.rungs}
        stale_ll = metrics.get("stale_only")
        raw_ll   = metrics.get("stale_plus_raw")
        emb_ll   = metrics.get("stale_plus_embedding")
        corr_ll  = metrics.get("stale_plus_corrupted")

        if not all([stale_ll, raw_ll, emb_ll]):
            return pd.DataFrame([{"note": "incomplete rungs"}])

        gain_raw   = stale_ll.log_loss - raw_ll.log_loss    # positive = improvement
        gain_embed = stale_ll.log_loss - emb_ll.log_loss
        retention  = gain_embed / gain_raw if abs(gain_raw) > 1e-6 else float("nan")

        return pd.DataFrame([{
            "stale_only_logloss":       round(stale_ll.log_loss, 5),
            "stale_plus_raw_logloss":   round(raw_ll.log_loss, 5),
            "stale_plus_embed_logloss": round(emb_ll.log_loss, 5),
            "stale_plus_corrupt_logloss": round(corr_ll.log_loss, 5) if corr_ll else None,
            "gain_raw_context":         round(gain_raw, 5),
            "gain_embedding":           round(gain_embed, 5),
            "compression_retention":    round(retention, 4),
            "note": "retention ≈ 1 supports the compression hypothesis",
        }])


@dataclass(frozen=True)
class MVPExperimentArtifacts:
    """Protocol artifacts that make the MVP experiment auditable from the public interface."""

    protocol_summary: pd.DataFrame
    split_summary: pd.DataFrame
    train_preview: pd.DataFrame
    test_preview: pd.DataFrame
    feature_blocks: pd.DataFrame
    model_registry: pd.DataFrame
    objective_registry: pd.DataFrame
    rung_plan: pd.DataFrame


# ---------------------------------------------------------------------------
# Main ladder class
# ---------------------------------------------------------------------------

class BeliefUpdatingMVPExperiment:
    """
    Run the legacy 4-rung MVP outcome experiment.

    Parameters
    ----------
    manifest:
        Output of BeliefUpdatingDatasetBuilder.
    config:
        Training configuration.

    Example
    -------
    ::

        experiment = BeliefUpdatingMVPExperiment(manifest, BeliefUpdatingMVPConfig(n_epochs=30))
        artifacts = experiment.artifacts()
        results = experiment.run()
        print(results.to_dataframe())
        print(results.compression_summary())
    """

    def __init__(self, manifest: BeliefUpdatingManifest, config: BeliefUpdatingMVPConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or BeliefUpdatingMVPConfig()
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def artifacts(self, *, preview_rows: int = 5) -> MVPExperimentArtifacts:
        """
        Return the split/features/models/objectives artifacts needed for notebook-style
        walkthroughs without reimplementing the protocol logic outside the class.
        """
        train_df, test_df = self._split()
        preview_cols = [
            "market_id",
            "end_date",
            "horizon_hours",
            "delta_hours_int",
            "n_siblings",
            "stale_yes_probability",
            "label",
        ]
        preview_cols = [col for col in preview_cols if col in self.manifest.examples.columns]
        return MVPExperimentArtifacts(
            protocol_summary=self.protocol_summary(),
            split_summary=self.split_summary(),
            train_preview=train_df[preview_cols].head(preview_rows).copy(),
            test_preview=test_df[preview_cols].head(preview_rows).copy(),
            feature_blocks=self.feature_block_table(),
            model_registry=self.model_registry(),
            objective_registry=self.objective_registry(),
            rung_plan=self.rung_plan(),
        )

    def protocol_summary(self) -> pd.DataFrame:
        """Return a compact protocol summary table."""
        return pd.DataFrame([{
            "protocol_name": "belief_updating_mvp_outcome",
            "target_column": "label",
            "target_meaning": "terminal binary outcome",
            "split_type": "grouped_out_of_time",
            "group_key": "split_group_id_or_market_id",
            "stale_features": len(STALE_FEATURE_NAMES),
            "context_features_per_sibling": len(CONTEXT_FEATURE_NAMES),
            "aggregated_context_features": len(AGG_CONTEXT_FEATURE_NAMES),
            "pytorch_loss": "BCEWithLogitsLoss",
            "report_metrics": "log_loss,brier,roc_auc",
        }])

    def split_summary(self) -> pd.DataFrame:
        """Describe the grouped out-of-time split used by the experiment."""
        train_df, test_df = self._split()
        return pd.DataFrame([
            {
                "split": "train",
                "rows": len(train_df),
                "markets": int(train_df["market_id"].nunique()),
                "families": int(train_df["family_id"].nunique()) if "family_id" in train_df.columns else 0,
                "end_date_min": train_df["end_date"].min(),
                "end_date_max": train_df["end_date"].max(),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "markets": int(test_df["market_id"].nunique()),
                "families": int(test_df["family_id"].nunique()) if "family_id" in test_df.columns else 0,
                "end_date_min": test_df["end_date"].min(),
                "end_date_max": test_df["end_date"].max(),
            },
        ])

    def feature_blocks(self) -> dict[str, list[str]]:
        """Expose the three feature blocks used across the four rungs."""
        return {
            "stale_only": list(STALE_FEATURE_NAMES),
            "stale_plus_raw": list(STALE_FEATURE_NAMES) + [c for c in AGG_CONTEXT_FEATURE_NAMES if c in self.manifest.examples.columns],
            "context_per_sibling": list(CONTEXT_FEATURE_NAMES),
        }

    def feature_block_table(self) -> pd.DataFrame:
        """Return a readable table of feature blocks by rung family."""
        blocks = self.feature_blocks()
        rows: list[dict[str, Any]] = []
        for block_name, columns in blocks.items():
            rows.append({
                "block": block_name,
                "n_columns": len(columns),
                "columns": ", ".join(columns),
            })
        return pd.DataFrame(rows)

    def model_registry(self) -> pd.DataFrame:
        """Return the models used in each rung family."""
        return pd.DataFrame([
            {
                "model_family": "tabular_baseline",
                "used_by_rungs": "stale_only, stale_plus_raw",
                "model": "Pipeline(SimpleImputer -> StandardScaler -> HistGradientBoostingClassifier)",
            },
            {
                "model_family": "set_encoder",
                "used_by_rungs": "stale_plus_embedding, stale_plus_corrupted",
                "model": "DeepSetsEncoder + BeliefUpdatingPredictor",
            },
        ])

    def objective_registry(self) -> pd.DataFrame:
        """Return the losses and evaluation metrics used by the MVP protocol."""
        return pd.DataFrame([
            {
                "stage": "sklearn_training",
                "used_by_rungs": "stale_only, stale_plus_raw",
                "objective": "HistGradientBoostingClassifier built-in classification loss",
            },
            {
                "stage": "pytorch_training",
                "used_by_rungs": "stale_plus_embedding, stale_plus_corrupted",
                "objective": "BCEWithLogitsLoss",
            },
            {
                "stage": "evaluation",
                "used_by_rungs": "all",
                "objective": "log_loss, brier, roc_auc",
            },
        ])

    def rung_plan(self) -> pd.DataFrame:
        """Describe the meaning of each rung before training starts."""
        return pd.DataFrame([
            {
                "rung": "stale_only",
                "input_block": "stale_only",
                "model_family": "tabular_baseline",
                "purpose": "lower bound using only stale-local target state",
            },
            {
                "rung": "stale_plus_raw",
                "input_block": "stale_plus_raw",
                "model_family": "tabular_baseline",
                "purpose": "test whether aggregated raw context contains signal",
            },
            {
                "rung": "stale_plus_embedding",
                "input_block": "stale_only + context_per_sibling",
                "model_family": "set_encoder",
                "purpose": "test whether a compact context embedding preserves raw-context value",
            },
            {
                "rung": "stale_plus_corrupted",
                "input_block": "stale_only + shuffled context_per_sibling",
                "model_family": "set_encoder",
                "purpose": "falsification control for contextual information",
            },
        ])

    def run(self, *, verbose: bool = True) -> BeliefUpdatingMVPResults:
        """Run all four rungs and return results."""
        train_df, test_df = self._split()
        if verbose:
            self._log_protocol_start(train_df, test_df)

        train_idx = list(train_df.index)
        test_idx  = list(test_df.index)

        results = BeliefUpdatingMVPResults(rungs=[], config=self.config)
        for rung_fn, rung_name in [
            (self._rung_stale_only,         "stale_only"),
            (self._rung_stale_plus_raw,     "stale_plus_raw"),
            (self._rung_stale_plus_embedding, "stale_plus_embedding"),
            (self._rung_stale_plus_corrupted, "stale_plus_corrupted"),
        ]:
            if verbose:
                self._log_rung_start(rung_name, train_df=train_df, test_df=test_df)
            t0 = time.perf_counter()
            probs, history = rung_fn(train_df, test_df, train_idx, test_idx, verbose=verbose)
            elapsed = time.perf_counter() - t0

            labels = test_df["label"].to_numpy(dtype=float)
            metrics = _compute_metrics(labels, probs)
            rung_result = BeliefUpdatingMVPRungMetrics(
                rung=rung_name,
                n_train=len(train_df),
                n_test=len(test_df),
                train_time_s=elapsed,
                training_history=history,
                **metrics,
            )
            results.rungs.append(rung_result)
            if verbose:
                self._log_rung_end(rung_result)

        return results

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------

    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strict out-of-time split ordered by end_date and grouped by target market."""
        examples = self.manifest.examples.copy()
        group_col = "split_group_id" if "split_group_id" in examples.columns else "market_id"
        groups = (
            examples.groupby(group_col, dropna=False)["end_date"]
            .max()
            .sort_values(kind="stable")
            .reset_index()
        )
        split_idx = max(1, int(len(groups) * (1.0 - self.config.test_fraction)))
        train_groups = set(groups.iloc[:split_idx][group_col].astype(str))
        test_groups = set(groups.iloc[split_idx:][group_col].astype(str))

        train_df = examples.loc[examples[group_col].astype(str).isin(train_groups)].copy()
        test_df = examples.loc[examples[group_col].astype(str).isin(test_groups)].copy()
        train_df = train_df.sort_values("end_date", kind="stable")
        test_df = test_df.sort_values("end_date", kind="stable")
        return train_df, test_df

    # ------------------------------------------------------------------
    # Rung 1: stale-only baseline
    # ------------------------------------------------------------------

    def _rung_stale_only(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_idx: list[int],
        test_idx: list[int],
        *,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Predict from stale-local features only.
        Baseline: how well can we do with an outdated snapshot of the target market?
        """
        X_train = train_df[STALE_FEATURE_NAMES].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=float)
        X_test  = test_df[STALE_FEATURE_NAMES].to_numpy(dtype=float)

        model = _build_gbm_pipeline(self.config)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1], []

    # ------------------------------------------------------------------
    # Rung 2: stale + aggregated raw context
    # ------------------------------------------------------------------

    def _rung_stale_plus_raw(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_idx: list[int],
        test_idx: list[int],
        *,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Predict from stale features + hand-aggregated (mean/max) context.
        Tests whether raw non-local context contains update-relevant signal,
        before any learned compression.
        """
        feat_cols = STALE_FEATURE_NAMES + AGG_CONTEXT_FEATURE_NAMES
        # Only use columns that exist in the DataFrame
        feat_cols = [c for c in feat_cols if c in train_df.columns]

        X_train = train_df[feat_cols].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=float)
        X_test  = test_df[feat_cols].to_numpy(dtype=float)

        model = _build_gbm_pipeline(self.config)
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1], []

    # ------------------------------------------------------------------
    # Rung 3: stale + compact DeepSets embedding
    # ------------------------------------------------------------------

    def _rung_stale_plus_embedding(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_idx: list[int],
        test_idx: list[int],
        *,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Predict from stale features + z_t learned by the DeepSets encoder.
        If Perf(rung3) ≈ Perf(rung2) >> Perf(rung1), the compact embedding
        preserves most of the raw-context value (compression claim).
        """
        return self._run_pytorch_rung(
            train_df=train_df,
            test_df=test_df,
            train_idx=train_idx,
            test_idx=test_idx,
            corrupt_context=False,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Rung 4: stale + corrupted (shuffled) context
    # ------------------------------------------------------------------

    def _rung_stale_plus_corrupted(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_idx: list[int],
        test_idx: list[int],
        *,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Predict from stale features + z̃_t (context shuffled across examples).
        If Perf(rung4) ≈ Perf(rung1), the gain in rung 3 comes from genuine
        contextual information, not from extra parameters alone.
        """
        return self._run_pytorch_rung(
            train_df=train_df,
            test_df=test_df,
            train_idx=train_idx,
            test_idx=test_idx,
            corrupt_context=True,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Shared PyTorch training loop (rungs 3 and 4)
    # ------------------------------------------------------------------

    def _run_pytorch_rung(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_idx: list[int],
        test_idx: list[int],
        corrupt_context: bool,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Train the DeepSets encoder end-to-end with BCE loss, then predict.

        corrupt_context=True: context is shuffled across examples in each
        mini-batch, breaking the link between context and target (rung 4).
        """
        cfg = self.config
        device = torch.device(cfg.device)

        # Build datasets
        train_dataset = BeliefUpdatingTorchDataset(
            self.manifest, indices=list(train_df.index), max_context_size=cfg.max_context_size,
        )
        test_dataset = BeliefUpdatingTorchDataset(
            self.manifest, indices=list(test_df.index), max_context_size=cfg.max_context_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=BeliefUpdatingTorchDataset.collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=BeliefUpdatingTorchDataset.collate_fn,
        )

        # Build model
        encoder = DeepSetsEncoder(
            input_dim=len(CONTEXT_FEATURE_NAMES),
            hidden_dim=cfg.encoder_hidden_dim,
            output_dim=cfg.encoder_output_dim,
        ).to(device)
        predictor = BeliefUpdatingPredictor(
            flat_dim=len(STALE_FEATURE_NAMES),
            context_dim=cfg.encoder_output_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=cfg.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()
        history: list[dict[str, float]] = []

        # Training loop
        for epoch in range(cfg.n_epochs):
            encoder.train()
            predictor.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                if corrupt_context:
                    batch = shuffle_context_across_batch(batch)
                flat    = batch["flat"].to(device)
                context = batch["context"].to(device)
                mask    = batch["mask"].to(device)
                labels  = batch["label"].to(device)

                optimizer.zero_grad()
                z = encoder(context, mask)
                logits = predictor(flat, z)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            epoch_mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_train_loss": epoch_mean_loss,
                    "n_batches": float(len(epoch_losses)),
                }
            )
            if verbose and cfg.show_epoch_progress:
                if (epoch + 1) == 1 or (epoch + 1) == cfg.n_epochs or ((epoch + 1) % max(1, cfg.log_every_epochs) == 0):
                    print(
                        f"    {_render_progress_bar(epoch + 1, cfg.n_epochs)} "
                        f"epoch {epoch + 1}/{cfg.n_epochs} "
                        f"mean_train_loss={epoch_mean_loss:.5f}"
                    )

        # Inference on test set
        encoder.eval()
        predictor.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                if corrupt_context:
                    batch = shuffle_context_across_batch(batch)
                flat    = batch["flat"].to(device)
                context = batch["context"].to(device)
                mask    = batch["mask"].to(device)
                z = encoder(context, mask)
                logits = predictor(flat, z)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0), history

    def _log_protocol_start(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Print a protocol summary before the run starts."""
        summary = self.protocol_summary().iloc[0].to_dict()
        print(f"[protocol] {summary['protocol_name']}")
        print(
            f"[split] train_rows={len(train_df)} test_rows={len(test_df)} "
            f"train_markets={train_df['market_id'].nunique()} test_markets={test_df['market_id'].nunique()}"
        )
        print(
            f"[features] stale={len(STALE_FEATURE_NAMES)} "
            f"context_item={len(CONTEXT_FEATURE_NAMES)} "
            f"aggregated_context={len([c for c in AGG_CONTEXT_FEATURE_NAMES if c in self.manifest.examples.columns])}"
        )
        print("[objective] target=label (terminal outcome) metrics=log_loss,brier,roc_auc")

    def _log_rung_start(self, rung_name: str, *, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Print a readable rung header."""
        plan = self.rung_plan().set_index("rung").to_dict(orient="index").get(rung_name, {})
        print(f"\n[{rung_name}] {plan.get('purpose', 'training')}")
        print(
            f"  input_block={plan.get('input_block', 'unknown')} "
            f"model_family={plan.get('model_family', 'unknown')} "
            f"train={len(train_df)} test={len(test_df)}"
        )

    def _log_rung_end(self, rung_result: BeliefUpdatingMVPRungMetrics) -> None:
        """Print a readable rung footer."""
        print(
            f"  finished in {rung_result.train_time_s:.1f}s | "
            f"log_loss={rung_result.log_loss:.4f} "
            f"brier={rung_result.brier:.4f} "
            f"roc_auc={rung_result.roc_auc:.4f}"
        )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_gbm_pipeline(config: BeliefUpdatingMVPConfig) -> Pipeline:
    """Build a sklearn pipeline: imputation → standardization → HistGBT."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  HistGradientBoostingClassifier(
            max_iter=config.gbm_max_iter,
            max_depth=4,
            learning_rate=0.05,
            random_state=config.random_state,
        )),
    ])


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute log-loss, Brier score, and ROC-AUC."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    ll = float(log_loss(labels, probs))
    brier = float(np.mean((probs - labels) ** 2))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {"log_loss": ll, "brier": brier, "roc_auc": auc}


def _render_progress_bar(current: int, total: int, *, width: int = 16) -> str:
    """Render a lightweight text progress bar."""
    if total <= 0:
        return "[----------------]"
    filled = min(width, int(round((current / total) * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

