"""
Belief Updating Model
=====================

Two PyTorch modules and a PyTorch Dataset for the contextual belief-updating task.

DeepSetsEncoder
    Encodes a *variable-size set* of sibling market states into a fixed-size
    embedding z_t ∈ R^{d_z}. Uses a shared per-element MLP followed by
    mean + max pooling (a DeepSets-style architecture). Permutation invariant
    by construction.

BeliefUpdatingPredictor
    Combines flat target-side features with the context embedding z_t and
    outputs a scalar prediction. The same head can be used for binary outcome
    classification or continuous update recovery.

BeliefUpdatingTorchDataset
    Wraps a BeliefUpdatingManifest and assembles per-example tensors on demand.
    Handles variable-length context by padding to max_context_size rows and
    producing a boolean padding mask.

Usage
-----
::

    from polymarket_research.belief_updating import (
        BeliefUpdatingTorchDataset, DeepSetsEncoder, BeliefUpdatingPredictor,
    )
    from torch.utils.data import DataLoader

    dataset = BeliefUpdatingTorchDataset(manifest, train_indices)
    loader  = DataLoader(dataset, batch_size=64, collate_fn=dataset.collate_fn)

    encoder   = DeepSetsEncoder(input_dim=len(CONTEXT_FEATURE_NAMES))
    predictor = BeliefUpdatingPredictor(
        stale_dim=len(STALE_FEATURE_NAMES),
        context_dim=encoder.output_dim,
    )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from polymarket_research.belief_updating.dataset import (
    CONTEXT_FEATURE_NAMES,
    STALE_FEATURE_NAMES,
    BeliefUpdatingManifest,
)


# ---------------------------------------------------------------------------
# Set encoder
# ---------------------------------------------------------------------------

class DeepSetsEncoder(nn.Module):
    """
    Permutation-invariant set encoder (DeepSets style).

    Architecture
    ------------
    1. Per-element MLP  : input_dim → hidden_dim → hidden_dim   (shared weights)
    2. Pooling          : concatenate( mean(H), max(H) )        → 2 * hidden_dim
    3. Projection MLP   : 2 * hidden_dim → output_dim

    The masked mean/max pooling ignores padded (zero-masked) rows so the
    encoding does not depend on the padding level.

    Parameters
    ----------
    input_dim:
        Dimensionality of each context item (= len(CONTEXT_FEATURE_NAMES)).
    hidden_dim:
        Width of the per-element and projection MLPs.
    output_dim:
        Dimensionality of the output embedding z_t.
    dropout:
        Dropout probability applied inside the per-element MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Per-element MLP (shared across all context items in the set)
        self.element_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Projection after mean+max pooling
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        context: torch.Tensor,   # (B, N, D_in)
        mask: torch.Tensor,       # (B, N) bool, True = valid item
    ) -> torch.Tensor:            # (B, output_dim)
        """
        Encode a batch of variable-size context sets.

        Parameters
        ----------
        context:
            Float tensor of shape (B, N, D_in).  Padded items can have any
            value; they are excluded by *mask*.
        mask:
            Boolean tensor of shape (B, N).  True indicates a valid (non-padded)
            context item.

        Returns
        -------
        Tensor of shape (B, output_dim).
        """
        B, N, _ = context.shape

        # Per-element encoding: apply the shared MLP to every item
        h = self.element_mlp(context)  # (B, N, hidden_dim)

        # Masked pooling: replace padded positions with −∞ (for max) or 0 (for mean)
        float_mask = mask.float().unsqueeze(-1)  # (B, N, 1)

        # Mean pool: sum valid items, divide by valid count
        h_mean = (h * float_mask).sum(dim=1)                     # (B, hidden_dim)
        valid_count = float_mask.sum(dim=1).clamp(min=1.0)       # (B, 1)
        h_mean = h_mean / valid_count

        # Max pool: set padded to large negative before max
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        h_max = h_for_max.max(dim=1).values                       # (B, hidden_dim)

        # Combine and project
        pooled = torch.cat([h_mean, h_max], dim=-1)               # (B, 2 * hidden_dim)
        z = self.projection(pooled)                                # (B, output_dim)
        return z


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class BeliefUpdatingPredictor(nn.Module):
    """
    Predicts a scalar target from flat features + context embedding.

    Architecture
    ------------
        [flat_features | context_embedding]  →  MLP  →  scalar prediction

    The encoder and predictor are separate modules so the encoder can be frozen
    during transfer probes.

    Parameters
    ----------
    stale_dim:
        Number of flat target-side features.
    context_dim:
        Dimensionality of the context embedding (= encoder.output_dim).
    hidden_dim:
        Hidden width of the prediction MLP.
    dropout:
        Dropout probability inside the MLP.
    """

    def __init__(
        self,
        flat_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.flat_dim = flat_dim
        self.context_dim = context_dim

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        flat: torch.Tensor,       # (B, flat_dim)
        context_z: torch.Tensor,  # (B, context_dim)
    ) -> torch.Tensor:            # (B,) scalar prediction
        """
        Concatenate flat features and context embedding, then predict.

        Returns a raw scalar. For classification use BCEWithLogitsLoss; for
        update recovery use MSELoss or SmoothL1Loss.
        """
        x = torch.cat([flat, context_z], dim=-1)  # (B, flat_dim + context_dim)
        return self.mlp(x).squeeze(-1)             # (B,)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BeliefUpdatingTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping a BeliefUpdatingManifest.

    For each example the dataset:
    1. Returns selected flat features as a 1-D float tensor.
    2. Assembles the context matrix by looking up sibling snapshots at the
       example's context_time and family, then stacks them row-wise.
    3. Pads the context matrix to *max_context_size* rows with zeros, and
       returns a boolean mask (True = valid row).

    The target-relative feature ``ctx_prob_vs_target_stale`` is computed on
    the fly (it depends on the stale probability of the target market).

    Parameters
    ----------
    manifest:
        Output of BeliefUpdatingDatasetBuilder.
    indices:
        Integer indices into manifest.examples to include.  Pass None to use all.
    max_context_size:
        Maximum number of siblings per example.  Longer sets are truncated
        (randomly on each access to add slight stochasticity).
    """

    def __init__(
        self,
        manifest: BeliefUpdatingManifest,
        indices: Sequence[int] | None = None,
        max_context_size: int = 16,
        flat_feature_names: Sequence[str] | None = None,
        label_col: str = "label",
    ) -> None:
        self.manifest = manifest
        self.max_context_size = max_context_size
        self.label_col = label_col

        examples = manifest.examples
        if indices is not None:
            examples = examples.iloc[list(indices)].reset_index(drop=True)
        self.examples = examples

        # Build a fast lookup: (market_id, snapshot_time) → feature vector
        # We index context_snapshots by (market_id, snapshot_time) for O(1) access.
        ctx = manifest.context_snapshots
        # Use columns that exist in context snapshots (without target-relative field)
        ctx_cols = [c for c in CONTEXT_FEATURE_NAMES if c != "ctx_prob_vs_target_stale"]
        self._ctx_lookup: dict[tuple, np.ndarray] = {}
        for row in ctx.itertuples(index=False):
            key = (str(row.market_id), row.snapshot_time)
            vec = np.array([getattr(row, col, np.nan) for col in ctx_cols], dtype=np.float32)
            self._ctx_lookup[key] = vec

        # Pre-build family map from context snapshots when available so valid
        # context-only siblings are not dropped just because they never become
        # target examples themselves.
        family_source = manifest.context_snapshots if "family_id" in manifest.context_snapshots.columns else manifest.examples
        family_col = "family_id" if "family_id" in family_source.columns else None
        self._family_map: dict[str, list[str]] = {}
        if family_col is not None:
            for row in family_source.itertuples(index=False):
                fid = str(row.family_id)
                mid = str(row.market_id)
                if fid not in self._family_map:
                    self._family_map[fid] = []
                if mid not in self._family_map[fid]:
                    self._family_map[fid].append(mid)

        self._flat_cols = list(flat_feature_names or manifest.stale_feature_names)
        # Context feature order: ctx_cols + [ctx_prob_vs_target_stale]
        self._ctx_base_cols = ctx_cols
        # The full CONTEXT_FEATURE_NAMES order determines output column order
        self._ctx_prob_idx = CONTEXT_FEATURE_NAMES.index("ctx_prob_vs_target_stale")
        self._ctx_feature_dim = len(CONTEXT_FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a dict with keys:
          flat     : float32 tensor, shape (D_flat,)
          context  : float32 tensor, shape (max_context_size, D_ctx)
          mask     : bool tensor,    shape (max_context_size,)
          label    : float32 scalar tensor
        """
        row = self.examples.iloc[idx]
        target_id = str(row["market_id"])
        family_id = str(row["family_id"])
        context_time = row["context_time"]
        stale_prob = float(row["stale_yes_probability"])

        # --- Stale features ---
        flat_vec = np.array(
            [float(row[col]) if col in row and not pd.isna(row[col]) else 0.0
             for col in self._flat_cols],
            dtype=np.float32,
        )
        # Replace NaN with 0 (imputation handled by training pipeline)
        flat_vec = np.nan_to_num(flat_vec, nan=0.0)

        # --- Context matrix ---
        # Siblings = all markets in the same family whose snapshots exist at context_time,
        # excluding the target itself.
        sibling_ids = self._family_map.get(family_id, [])
        ctx_rows: list[np.ndarray] = []
        for sib_id in sibling_ids:
            if sib_id == target_id:
                continue
            key = (sib_id, context_time)
            base_vec = self._ctx_lookup.get(key)
            if base_vec is None:
                continue
            # Construct full context vector including target-relative feature
            full_vec = np.empty(self._ctx_feature_dim, dtype=np.float32)
            full_vec[:self._ctx_prob_idx] = base_vec[:self._ctx_prob_idx]
            full_vec[self._ctx_prob_idx] = base_vec[0] - stale_prob  # ctx_prob_vs_target_stale
            full_vec[self._ctx_prob_idx + 1:] = base_vec[self._ctx_prob_idx:]
            full_vec = np.nan_to_num(full_vec, nan=0.0)
            ctx_rows.append(full_vec)

        # Pad or truncate to max_context_size
        n_valid = min(len(ctx_rows), self.max_context_size)
        context_mat = np.zeros((self.max_context_size, self._ctx_feature_dim), dtype=np.float32)
        mask = np.zeros(self.max_context_size, dtype=bool)
        if n_valid > 0:
            context_mat[:n_valid] = np.stack(ctx_rows[:n_valid], axis=0)
            mask[:n_valid] = True

        return {
            "flat": torch.tensor(flat_vec, dtype=torch.float32),
            "context": torch.tensor(context_mat, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Stack a list of single-example dicts into batched tensors."""
        return {
            "flat":    torch.stack([b["flat"] for b in batch]),     # (B, D_flat)
            "context": torch.stack([b["context"] for b in batch]),  # (B, N, D_ctx)
            "mask":    torch.stack([b["mask"] for b in batch]),     # (B, N)
            "label":   torch.stack([b["label"] for b in batch]),    # (B,)
        }


# ---------------------------------------------------------------------------
# Corruption utility
# ---------------------------------------------------------------------------

def shuffle_context_across_batch(batch: dict) -> dict:
    """
    Corrupt context by shuffling context matrices across examples in the batch.

    This produces the ``z̃_t`` corrupted-context control (rung 4 in the paper).
    The shuffled assignment is random per batch and does not preserve any
    semantic alignment between the stale target state and the context.

    Returns a copy of the batch dict with the context and mask shuffled.
    """
    B = batch["context"].shape[0]
    perm = torch.randperm(B)
    return {
        "flat":    batch["flat"],
        "context": batch["context"][perm],
        "mask":    batch["mask"][perm],
        "label":   batch["label"],
    }
