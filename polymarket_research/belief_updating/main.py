"""
Main belief-update recovery experiment.

This module implements the stricter protocol described in the repository-level
``experiment.md``: the primary target is the hidden current-state update of the
target market at context time t, not its terminal outcome.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from polymarket_research.belief_updating.dataset import (
    AGG_CONTEXT_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    STALE_FEATURE_NAMES,
    BeliefUpdatingManifest,
)
from polymarket_research.belief_updating.model import (
    BeliefUpdatingPredictor,
    BeliefUpdatingTorchDataset,
    DeepSetsEncoder,
    shuffle_context_across_batch,
)


MAIN_FLAT_FEATURE_NAMES: list[str] = list(STALE_FEATURE_NAMES) + list(GLOBAL_FEATURE_NAMES)


@dataclass
class BeliefUpdatingMainConfig:
    """Training configuration for the main belief-update recovery experiment."""

    test_fraction: float = 0.25
    n_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    encoder_hidden_dim: int = 64
    large_encoder_output_dim: int = 128
    compact_encoder_output_dim: int = 32
    max_context_size: int = 16
    device: str = "cpu"
    random_state: int = 42
    gbm_max_iter: int = 300
    use_global_features: bool = True
    log_every_epochs: int = 1
    show_epoch_progress: bool = True


@dataclass
class BeliefUpdatingMainRungMetrics:
    """Metrics for one rung of the main belief-update experiment."""

    rung: str
    n_train: int
    n_test: int
    update_mae: float
    update_rmse: float
    update_r2: float
    current_prob_brier: float
    current_prob_mae: float
    train_time_s: float
    notes: str = ""
    training_history: list[dict[str, float]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "update_mae": round(self.update_mae, 5),
            "update_rmse": round(self.update_rmse, 5),
            "update_r2": round(self.update_r2, 5),
            "current_prob_brier": round(self.current_prob_brier, 5),
            "current_prob_mae": round(self.current_prob_mae, 5),
            "train_time_s": round(self.train_time_s, 1),
            "notes": self.notes,
        }


@dataclass
class BeliefUpdatingMainResults:
    """Collected metrics across all main-protocol rungs."""

    rungs: list[BeliefUpdatingMainRungMetrics]
    config: BeliefUpdatingMainConfig

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
        """Summarize how much of the large-context gain the compact model retains."""
        metrics = {r.rung: r for r in self.rungs}
        stale = metrics.get("stale_only")
        raw = metrics.get("stale_plus_raw")
        large = metrics.get("stale_plus_large_embedding")
        compact = metrics.get("stale_plus_compact_embedding")
        corrupt = metrics.get("stale_plus_corrupted")
        oracle = metrics.get("current_local_oracle")
        if not all([stale, raw, large, compact]):
            return pd.DataFrame([{"note": "incomplete rungs"}])

        gain_raw = stale.update_rmse - raw.update_rmse
        gain_large = stale.update_rmse - large.update_rmse
        gain_compact = stale.update_rmse - compact.update_rmse
        retention = gain_compact / gain_large if abs(gain_large) > 1e-6 else float("nan")
        return pd.DataFrame([{
            "stale_only_rmse": round(stale.update_rmse, 5),
            "stale_plus_raw_rmse": round(raw.update_rmse, 5),
            "stale_plus_large_rmse": round(large.update_rmse, 5),
            "stale_plus_compact_rmse": round(compact.update_rmse, 5),
            "stale_plus_corrupt_rmse": round(corrupt.update_rmse, 5) if corrupt else None,
            "current_local_oracle_rmse": round(oracle.update_rmse, 5) if oracle else None,
            "gain_raw": round(gain_raw, 5),
            "gain_large": round(gain_large, 5),
            "gain_compact": round(gain_compact, 5),
            "compression_retention_vs_large": round(retention, 4),
            "note": "retention close to 1 means the compact embedding preserves most of the large-context gain",
        }])


@dataclass(frozen=True)
class MainExperimentArtifacts:
    """Protocol artifacts that make the main experiment auditable from the public interface."""

    protocol_summary: pd.DataFrame
    split_summary: pd.DataFrame
    train_preview: pd.DataFrame
    test_preview: pd.DataFrame
    feature_blocks: pd.DataFrame
    model_registry: pd.DataFrame
    objective_registry: pd.DataFrame
    rung_plan: pd.DataFrame


class BeliefUpdatingMainExperiment:
    """Run the main belief-update recovery experiment."""

    def __init__(self, manifest: BeliefUpdatingManifest, config: BeliefUpdatingMainConfig | None = None) -> None:
        self.manifest = manifest
        self.config = config or BeliefUpdatingMainConfig()
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

    def artifacts(self, *, preview_rows: int = 5) -> MainExperimentArtifacts:
        """Return the split/features/models/objectives artifacts for notebook walkthroughs."""
        train_df, test_df = self._split()
        preview_cols = [
            "market_id",
            "end_date",
            "horizon_hours",
            "delta_hours_int",
            "n_siblings",
            "stale_yes_probability",
            "target_current_probability",
            "label_update_logit",
        ]
        preview_cols = [col for col in preview_cols if col in self.manifest.examples.columns]
        return MainExperimentArtifacts(
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
        """Return a compact summary of the main protocol."""
        return pd.DataFrame([{
            "protocol_name": "belief_updating_main_recovery",
            "target_column": "label_update_logit",
            "target_meaning": "hidden current-state update in logit space",
            "split_type": "grouped_out_of_time",
            "group_key": "split_group_id_or_market_id",
            "flat_features": len(self._flat_feature_cols()),
            "context_features_per_sibling": len(CONTEXT_FEATURE_NAMES),
            "aggregated_context_features": len([c for c in AGG_CONTEXT_FEATURE_NAMES if c in self.manifest.examples.columns]),
            "pytorch_loss": "MSELoss",
            "report_metrics": "update_mae,update_rmse,update_r2,current_prob_brier,current_prob_mae",
        }])

    def split_summary(self) -> pd.DataFrame:
        """Describe the grouped out-of-time split used by the main experiment."""
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
        """Expose the feature blocks used by the main protocol."""
        return {
            "flat_main_features": self._flat_feature_cols(),
            "stale_plus_raw": self._flat_feature_cols() + [c for c in AGG_CONTEXT_FEATURE_NAMES if c in self.manifest.examples.columns],
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
        """Return the models used in the main rung families."""
        return pd.DataFrame([
            {
                "model_family": "tabular_regression_baseline",
                "used_by_rungs": "stale_only, stale_plus_raw",
                "model": "Pipeline(SimpleImputer -> StandardScaler -> HistGradientBoostingRegressor)",
            },
            {
                "model_family": "set_encoder_regression",
                "used_by_rungs": "stale_plus_large_embedding, stale_plus_compact_embedding, stale_plus_corrupted",
                "model": "DeepSetsEncoder + BeliefUpdatingPredictor",
            },
            {
                "model_family": "oracle",
                "used_by_rungs": "current_local_oracle",
                "model": "read the true hidden current-state update",
            },
        ])

    def objective_registry(self) -> pd.DataFrame:
        """Return the losses and evaluation metrics used by the main protocol."""
        return pd.DataFrame([
            {
                "stage": "sklearn_training",
                "used_by_rungs": "stale_only, stale_plus_raw",
                "objective": "HistGradientBoostingRegressor squared-error objective",
            },
            {
                "stage": "pytorch_training",
                "used_by_rungs": "stale_plus_large_embedding, stale_plus_compact_embedding, stale_plus_corrupted",
                "objective": "MSELoss on label_update_logit",
            },
            {
                "stage": "evaluation",
                "used_by_rungs": "all",
                "objective": "update_mae, update_rmse, update_r2, current_prob_brier, current_prob_mae",
            },
        ])

    def rung_plan(self) -> pd.DataFrame:
        """Describe the meaning of each rung before training starts."""
        return pd.DataFrame([
            {
                "rung": "stale_only",
                "input_block": "flat_main_features",
                "model_family": "tabular_regression_baseline",
                "purpose": "lower bound using only stale-local and optional global features",
            },
            {
                "rung": "stale_plus_raw",
                "input_block": "stale_plus_raw",
                "model_family": "tabular_regression_baseline",
                "purpose": "test whether aggregated raw context contains update signal",
            },
            {
                "rung": "stale_plus_large_embedding",
                "input_block": "flat_main_features + context_per_sibling",
                "model_family": "set_encoder_regression",
                "purpose": "test a larger latent bottleneck before compression",
            },
            {
                "rung": "stale_plus_compact_embedding",
                "input_block": "flat_main_features + context_per_sibling",
                "model_family": "set_encoder_regression",
                "purpose": "test whether a compact latent state preserves most of the large-context value",
            },
            {
                "rung": "stale_plus_corrupted",
                "input_block": "flat_main_features + shuffled context_per_sibling",
                "model_family": "set_encoder_regression",
                "purpose": "falsification control for contextual information",
            },
            {
                "rung": "current_local_oracle",
                "input_block": "true hidden current state",
                "model_family": "oracle",
                "purpose": "upper bound that is not a fair deployable competitor",
            },
        ])

    def run(self, *, verbose: bool = True) -> BeliefUpdatingMainResults:
        train_df, test_df = self._split()
        if verbose:
            self._log_protocol_start(train_df, test_df)

        results = BeliefUpdatingMainResults(rungs=[], config=self.config)
        rung_defs = [
            ("stale_only", self._rung_stale_only),
            ("stale_plus_raw", self._rung_stale_plus_raw),
            ("stale_plus_large_embedding", self._rung_stale_plus_large_embedding),
            ("stale_plus_compact_embedding", self._rung_stale_plus_compact_embedding),
            ("stale_plus_corrupted", self._rung_stale_plus_corrupted),
            ("current_local_oracle", self._rung_current_local_oracle),
        ]

        for rung_name, rung_fn in rung_defs:
            if verbose:
                self._log_rung_start(rung_name, train_df=train_df, test_df=test_df)
            t0 = time.perf_counter()
            preds, history = rung_fn(train_df, test_df, verbose=verbose)
            elapsed = time.perf_counter() - t0
            metrics = _compute_update_metrics(
                update_true=test_df["label_update_logit"].to_numpy(dtype=float),
                update_pred=preds["update_pred"],
                current_prob_true=test_df["target_current_probability"].to_numpy(dtype=float),
                current_prob_pred=preds["current_prob_pred"],
            )
            rung_metrics = BeliefUpdatingMainRungMetrics(
                rung=rung_name,
                n_train=len(train_df),
                n_test=len(test_df),
                train_time_s=elapsed,
                training_history=history,
                **metrics,
            )
            results.rungs.append(rung_metrics)
            if verbose:
                self._log_rung_end(rung_metrics)

        return results

    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strict out-of-time split grouped by target market."""
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

    def _flat_feature_cols(self) -> list[str]:
        cols = list(STALE_FEATURE_NAMES)
        if self.config.use_global_features:
            cols.extend(name for name in GLOBAL_FEATURE_NAMES if name in self.manifest.examples.columns)
        return cols

    def _rung_stale_only(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        feat_cols = self._flat_feature_cols()
        return self._run_regressor(train_df, test_df, feature_cols=feat_cols), []

    def _rung_stale_plus_raw(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        feat_cols = self._flat_feature_cols() + [c for c in AGG_CONTEXT_FEATURE_NAMES if c in train_df.columns]
        return self._run_regressor(train_df, test_df, feature_cols=feat_cols), []

    def _rung_stale_plus_large_embedding(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        return self._run_pytorch_rung(
            train_df,
            test_df,
            output_dim=self.config.large_encoder_output_dim,
            corrupt_context=False,
            verbose=verbose,
        )

    def _rung_stale_plus_compact_embedding(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        return self._run_pytorch_rung(
            train_df,
            test_df,
            output_dim=self.config.compact_encoder_output_dim,
            corrupt_context=False,
            verbose=verbose,
        )

    def _rung_stale_plus_corrupted(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        return self._run_pytorch_rung(
            train_df,
            test_df,
            output_dim=self.config.compact_encoder_output_dim,
            corrupt_context=True,
            verbose=verbose,
        )

    def _rung_current_local_oracle(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        return ({
            "update_pred": test_df["label_update_logit"].to_numpy(dtype=float),
            "current_prob_pred": test_df["target_current_probability"].to_numpy(dtype=float),
        }, [])

    def _run_regressor(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature_cols: list[str],
    ) -> dict[str, np.ndarray]:
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["label_update_logit"].to_numpy(dtype=float)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        model = _build_regression_pipeline(self.config)
        model.fit(X_train, y_train)
        update_pred = model.predict(X_test).astype(float)
        current_prob_pred = _reconstruct_current_probs(
            stale_logit=test_df["target_stale_logit"].to_numpy(dtype=float),
            update_pred=update_pred,
        )
        return {"update_pred": update_pred, "current_prob_pred": current_prob_pred}

    def _run_pytorch_rung(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        output_dim: int,
        corrupt_context: bool,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        cfg = self.config
        device = torch.device(cfg.device)
        flat_cols = self._flat_feature_cols()

        train_dataset = BeliefUpdatingTorchDataset(
            self.manifest,
            indices=list(train_df.index),
            max_context_size=cfg.max_context_size,
            flat_feature_names=flat_cols,
            label_col="label_update_logit",
        )
        test_dataset = BeliefUpdatingTorchDataset(
            self.manifest,
            indices=list(test_df.index),
            max_context_size=cfg.max_context_size,
            flat_feature_names=flat_cols,
            label_col="label_update_logit",
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

        encoder = DeepSetsEncoder(
            input_dim=len(CONTEXT_FEATURE_NAMES),
            hidden_dim=cfg.encoder_hidden_dim,
            output_dim=output_dim,
        ).to(device)
        predictor = BeliefUpdatingPredictor(
            flat_dim=len(flat_cols),
            context_dim=output_dim,
        ).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=cfg.learning_rate,
        )
        criterion = nn.MSELoss()
        history: list[dict[str, float]] = []

        for epoch in range(cfg.n_epochs):
            encoder.train()
            predictor.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                if corrupt_context:
                    batch = shuffle_context_across_batch(batch)
                flat = batch["flat"].to(device)
                context = batch["context"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                z = encoder(context, mask)
                pred = predictor(flat, z)
                loss = criterion(pred, labels)
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

        encoder.eval()
        predictor.eval()
        update_preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                if corrupt_context:
                    batch = shuffle_context_across_batch(batch)
                flat = batch["flat"].to(device)
                context = batch["context"].to(device)
                mask = batch["mask"].to(device)
                z = encoder(context, mask)
                pred = predictor(flat, z)
                update_preds.append(pred.cpu().numpy())

        update_pred = np.concatenate(update_preds, axis=0)
        current_prob_pred = _reconstruct_current_probs(
            stale_logit=test_df["target_stale_logit"].to_numpy(dtype=float),
            update_pred=update_pred,
        )
        return {"update_pred": update_pred, "current_prob_pred": current_prob_pred}, history

    def _log_protocol_start(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Print a readable protocol summary before the run starts."""
        summary = self.protocol_summary().iloc[0].to_dict()
        print(f"[protocol] {summary['protocol_name']}")
        print(
            f"[split] train_rows={len(train_df)} test_rows={len(test_df)} "
            f"train_markets={train_df['market_id'].nunique()} test_markets={test_df['market_id'].nunique()}"
        )
        print(
            f"[features] flat={len(self._flat_feature_cols())} "
            f"context_item={len(CONTEXT_FEATURE_NAMES)} "
            f"aggregated_context={len([c for c in AGG_CONTEXT_FEATURE_NAMES if c in self.manifest.examples.columns])}"
        )
        print("[objective] target=label_update_logit metrics=update_mae,update_rmse,update_r2,current_prob_brier,current_prob_mae")

    def _log_rung_start(self, rung_name: str, *, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Print a readable rung header."""
        plan = self.rung_plan().set_index("rung").to_dict(orient="index").get(rung_name, {})
        print(f"\n[{rung_name}] {plan.get('purpose', 'running')}")
        print(
            f"  input_block={plan.get('input_block', 'unknown')} "
            f"model_family={plan.get('model_family', 'unknown')} "
            f"train={len(train_df)} test={len(test_df)}"
        )

    def _log_rung_end(self, rung_result: BeliefUpdatingMainRungMetrics) -> None:
        """Print a readable rung footer."""
        print(
            f"  finished in {rung_result.train_time_s:.1f}s | "
            f"update_rmse={rung_result.update_rmse:.4f} "
            f"current_prob_brier={rung_result.current_prob_brier:.4f} "
            f"update_r2={rung_result.update_r2:.4f}"
        )


def _build_regression_pipeline(config: BeliefUpdatingMainConfig) -> Pipeline:
    """Build the tabular baseline pipeline for update regression."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            max_iter=config.gbm_max_iter,
            max_depth=4,
            learning_rate=0.05,
            random_state=config.random_state,
        )),
    ])


def _reconstruct_current_probs(*, stale_logit: np.ndarray, update_pred: np.ndarray) -> np.ndarray:
    """Map predicted updates back to current probability space."""
    logits = np.asarray(stale_logit, dtype=float) + np.asarray(update_pred, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def _compute_update_metrics(
    *,
    update_true: np.ndarray,
    update_pred: np.ndarray,
    current_prob_true: np.ndarray,
    current_prob_pred: np.ndarray,
) -> dict[str, float]:
    """Compute the main-protocol regression and reconstruction metrics."""
    update_true = np.asarray(update_true, dtype=float)
    update_pred = np.asarray(update_pred, dtype=float)
    current_prob_true = np.clip(np.asarray(current_prob_true, dtype=float), 1e-6, 1.0 - 1e-6)
    current_prob_pred = np.clip(np.asarray(current_prob_pred, dtype=float), 1e-6, 1.0 - 1e-6)

    err = update_pred - update_true
    prob_err = current_prob_pred - current_prob_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((update_true - np.mean(update_true)) ** 2))
    update_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {
        "update_mae": float(np.mean(np.abs(err))),
        "update_rmse": float(np.sqrt(np.mean(err ** 2))),
        "update_r2": update_r2,
        "current_prob_brier": float(np.mean(prob_err ** 2)),
        "current_prob_mae": float(np.mean(np.abs(prob_err))),
    }


def _render_progress_bar(current: int, total: int, *, width: int = 16) -> str:
    """Render a lightweight text progress bar."""
    if total <= 0:
        return "[----------------]"
    filled = min(width, int(round((current / total) * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
