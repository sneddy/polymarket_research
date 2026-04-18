Below is the version I would actually implement for the final paper. It keeps the core idea in your draft, but makes the experiment much cleaner: the main task becomes **masked current-state recovery**, and terminal outcome / repricing / trust become **transfer probes**. That is much closer to the paper’s stated “belief updating” claim than the current outcome-only primary task in the PDF. 

## 1. Final experiment protocol

### 1.1 Main claim

For a target market (A), at context time (t), the model sees:

* a **stale local view** of the target at (t_s = t-\Delta),
* a **fresh non-local context set** of related markets at (t),
* optional **global exogenous features** at (t) such as BTC/ETH returns,

but it does **not** see the target’s own current state at (t).

The primary target should be the hidden update:

[
\Delta \ell_A(t) = \operatorname{logit}(p_A(t)) - \operatorname{logit}(p_A(t_s))
]

where (p_A(t)) is the target market’s current probability at time (t).

The model predicts (\widehat{\Delta \ell}_A(t)), then reconstructs

[
\widehat{\ell}_A(t)=\operatorname{logit}(p_A(t_s)) + \widehat{\Delta \ell}_A(t),
\quad
\widehat{p}_A(t)=\sigma(\widehat{\ell}_A(t)).
]

That is the clean belief-update object.

### 1.2 Market universe

Use all **resolved binary markets** from Polymarket and Kalshi with 5-minute probability panels. BTC/ETH should be included as simple global numeric covariates in the main experiment. GDELT should stay out of the main claim and appear only as an appendix ablation.

Each market needs:

* platform
* market id
* question / title / description
* tags / category
* created time
* close / resolution time
* final binary outcome
* 5-minute panel with probability and activity fields

### 1.3 Two relation systems

You need two different relation graphs.

**Duplicate clusters.** Same underlying event, often cross-platform.
Example: the Kalshi and Polymarket versions of the same contract.

**Family clusters.** Related but not identical markets.
Example: different election submarkets within the same race.

For the **primary result**, context should include **family members but exclude exact duplicates**. Exact duplicates are too informative and should be treated as an upper-bound ablation, not the main result.

Also exclude any sibling whose market was already resolved at time (t) in the primary result.

### 1.4 Example generation

For each target market (A), choose horizons (H \in {24,72,168}) hours before resolution and staleness gaps (\Delta \in {24,72,168}).

Define:

[
t = \text{resolve_time}(A) - H,
\qquad
t_s = t - \Delta.
]

Keep the example only if all of the following hold:

1. (t_s > \text{created_at}(A))
2. target has a valid stale snapshot at or before (t_s)
3. target has a valid current snapshot at or before (t) for labeling
4. at least one usable context market exists

Use a maximum snapshot age threshold, for example 12 hours, exactly as in the draft.

For each retained example, build:

**Stale target features** (x^{\text{stale}}_A)

* stale probability or stale logit
* confidence margin (|p-0.5|)
* 24h and 168h price change
* 24h and 168h volatility
* 24h and 168h activity / trade count features
* life progress
* hours to resolution
* staleness gap (\Delta)

**Context element features** (c_i) for each sibling (i) at time (t)

* current sibling probability or logit
* confidence margin
* 24h and 168h price change
* volatility
* activity / trade count
* life progress
* hours to sibling resolution
* divergence from target stale state, e.g.
  [
  \operatorname{logit}(p_i(t)) - \operatorname{logit}(p_A(t_s))
  ]
* platform id
* relation type: family or duplicate

**Global features** (g_t)

* BTC return over 1h / 24h
* ETH return over 1h / 24h
* optional shock flags

**Labels**

* primary: (y_{\Delta} = \Delta \ell_A(t))
* also store (y_{\text{current}} = \ell_A(t))
* downstream: terminal outcome
* downstream: repricing label, e.g. whether (|p_A(t+24h)-p_A(t)| \ge 0.15)
* downstream: stale error label, e.g. (|p_A(t)-p_A(t_s)|)

### 1.5 Split protocol

Split must be both **out-of-time** and **cluster-aware**.

Sort targets by resolution time. Then create train / val / test, for example 70 / 10 / 20. The grouping key must be:

* duplicate cluster id, if available
* otherwise target market id

That means all examples from the same target market, and all exact cross-platform duplicates of that target, stay on the same side of the split.

This is stricter than the current draft and fixes a major review risk.

### 1.6 Main comparisons

The cleanest ladder is this:

1. **Stale-only**
   Predict update from stale features only.

2. **Stale + raw aggregated context**
   Mean / max / count summary of context, no learned set encoder.

3. **Stale + learned uncompressed context**
   Learned set encoder with a large bottleneck, e.g. (d_z=128).

4. **Stale + learned compact context**
   Same architecture, small bottleneck, e.g. (d_z=16) or (32).

5. **Stale + corrupted context**
   Same architecture as 4, but permute context sets across examples.

6. **Current-local oracle**
   Use the target’s true current state at (t). This is not a fair competitor; it is an upper bound.

For paper cleanliness, use the **same residual prediction head** for 1–5. You can still report GBDT baselines, but make them auxiliary, not the core compression ladder.

### 1.7 Metrics

For the primary task, report:

* MAE or RMSE on (\Delta \ell)
* Brier score between (\widehat{p}_A(t)) and (p_A(t))
* correlation or (R^2) on current logit recovery

I would define two summary quantities:

[
\text{GapClosed} = \frac{L_{\text{stale}} - L_{\text{model}}}{L_{\text{stale}}}
]

and

[
\text{Retention} = \frac{L_{\text{stale}} - L_{\text{compact}}}
{L_{\text{stale}} - L_{\text{large}}}
]

where (L) is the primary loss. This makes compression much cleaner than comparing the compact model only to a tabular aggregated baseline.

Then use frozen or lightly tuned probes on (z_t) for:

* terminal outcome prediction
* repricing prediction
* trust / stale-error prediction

Report hard slices:

* long staleness
* uncertain stale prior
* low activity
* rich family
* cross-platform family available vs not available

Use bootstrap confidence intervals grouped by duplicate cluster or target market, not by row.

---

## 2. Model

I would use a **target-conditioned set encoder**. It is only a small step beyond DeepSets, but it matches the task much better.

Let (x_s) be stale target features, (C_t={c_1,\dots,c_N}) the context set, and (g_t) the global features.

### 2.1 Stale encoder

[
s = \mathrm{MLP}_{\text{stale}}(x_s) \in \mathbb{R}^{d_h}
]

### 2.2 Context element encoder

For each context element (c_i),

[
e_i = \mathrm{MLP}*{\text{ctx}}(c_i) + E*{\text{platform}}(q_i) + E_{\text{rel}}(r_i)
]

where (q_i) is platform id and (r_i) is relation type.

### 2.3 Target-conditioned pooling

Compute attention weights conditioned on the stale target representation:

[
a_i \propto \exp\left(v^\top \tanh(W_s s + W_e e_i)\right)
]

Then pool as

[
u = \left[
\sum_i a_i e_i ; | ; \operatorname{mean}_i(e_i) ; | ; \operatorname{max}_i(e_i)
\right]
]

### 2.4 Compact market-state representation

[
z_t = \mathrm{MLP}_{z}(u) \in \mathbb{R}^{d_z}
]

This is the compressed non-local market state.

### 2.5 Residual update head

Predict the hidden belief update, not the whole current state from scratch:

[
\widehat{\Delta \ell}*A(t) = \mathrm{MLP}*{\Delta}([s | z_t | g_t])
]

[
\widehat{\ell}_A(t)=\ell_A(t_s)+\widehat{\Delta \ell}_A(t)
]

Optionally add an auxiliary head for terminal outcome:

[
\widehat{y}*A = \sigma(\mathrm{MLP}*{\text{out}}([s | z_t | g_t]))
]

### 2.6 Loss

Primary loss:

[
\mathcal{L}_{\text{primary}} = \mathrm{Huber}(\widehat{\Delta \ell}, \Delta \ell)
]

Optional auxiliary loss:

[
\mathcal{L} = \mathcal{L}_{\text{primary}} + \lambda , \mathrm{BCE}(\widehat{y}, y)
]

with (\lambda) small, such as 0.1.

A good default is (d_h=64), (d_z \in {16,32,64}), dropout 0.1.

---

## 3. Training process protocol

### 3.1 Offline data build

Before training anything, freeze these artifacts:

* `markets.parquet`
* `price_panels.parquet`
* `relations.parquet` with family and duplicate ids
* `example_manifest.parquet`
* `context_rows.parquet`
* `splits.parquet`
* `config.yaml`

The manifest should hold one row per example. The context table should hold one row per example-context pair.

### 3.2 Feature normalization

Normalize numeric stale and global features using train-set statistics only.
For context features, either normalize before storage or normalize in the dataset loader using train stats.

### 3.3 Primary training

Train the compact and uncompressed neural models on the primary update target.

Good defaults:

* optimizer: AdamW
* learning rate: (10^{-3})
* weight decay: (10^{-4})
* batch size: 256
* epochs: 50 max
* early stopping: patience 5 on validation primary loss
* gradient clipping: 1.0
* seeds: at least 3 runs

The corrupted-context control should use the exact same code path and hyperparameters, except that context sets are permuted within batch.

### 3.4 Transfer probes

After primary training:

1. freeze the encoder
2. extract (z_t) for train / val / test
3. fit probes for:

   * outcome
   * repricing
   * stale error / trust

Use both:

* a **linear probe**
* a **small MLP probe**

If the linear probe already works, the representation is much easier to defend.

### 3.5 Integrity checks

Implement these as hard assertions, not comments:

* target market never appears in its own context
* no source timestamp is after the relevant cutoff
* no duplicate cluster crosses split boundaries
* no resolved-before-(t) siblings in the primary result
* corrupted-context run preserves context-size distribution

---

## 4. Coding interfaces to implement

The cleanest way is to make the experiment modular around five layers:

1. canonical data access
2. relation building
3. example building
4. model + adapters
5. training / evaluation

### 4.1 Core configs and schemas

```python
# config.py
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class ExperimentConfig:
    horizons_h: tuple[int, ...] = (24, 72, 168)
    staleness_h: tuple[int, ...] = (24, 72, 168)
    max_snapshot_age_h: int = 12
    include_duplicate_context: bool = False
    include_resolved_siblings: bool = False
    max_context_size: int = 32

    hidden_dim: int = 64
    compact_dim: int = 32
    large_dim: int = 128
    dropout: float = 0.10

    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5
    aux_outcome_weight: float = 0.1

    split_train: float = 0.70
    split_val: float = 0.10
    random_seed: int = 42
```

```python
# schema.py
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

Platform = Literal["polymarket", "kalshi"]
RelationType = Literal["family", "duplicate"]
Split = Literal["train", "val", "test"]

@dataclass(frozen=True)
class ExampleKey:
    target_market_id: str
    context_ts: datetime
    stale_ts: datetime
    horizon_h: int
    staleness_h: int

@dataclass(frozen=True)
class ExampleRow:
    example_id: str
    key: ExampleKey
    target_platform: Platform
    family_id: str
    duplicate_cluster_id: Optional[str]
    split: Split

    y_delta_logit: float
    y_current_logit: float
    y_outcome: int
    y_reprice_24h: int

@dataclass(frozen=True)
class ContextRow:
    example_id: str
    sibling_market_id: str
    sibling_platform: Platform
    relation_type: RelationType
    resolved_before_context: bool
```

On disk, these should be flat Parquet tables. In memory, you can map them into tensors.

### 4.2 Canonical data and feature providers

```python
# data_interfaces.py
from typing import Protocol, Optional
from datetime import datetime
import pandas as pd

class CanonicalStore(Protocol):
    def markets(self) -> pd.DataFrame: ...
    def snapshots(self, market_id: str, start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> pd.DataFrame: ...
    def latest_snapshot_at_or_before(self, market_id: str, ts: datetime,
                                     max_age_h: int) -> Optional[pd.Series]: ...

class GlobalFeatureProvider(Protocol):
    def features_at(self, ts: datetime) -> dict[str, float]: ...
```

`CanonicalStore` is the single most important interface. Everything downstream should read from it, not from raw ad hoc files.

### 4.3 Relation building

```python
# relations.py
from typing import Protocol
import pandas as pd

class RelationBuilder(Protocol):
    def build_family_index(self, markets: pd.DataFrame) -> pd.DataFrame: ...
    def build_duplicate_index(self, markets: pd.DataFrame) -> pd.DataFrame: ...

class RelationIndex(Protocol):
    def family_members(self, market_id: str) -> list[str]: ...
    def duplicate_members(self, market_id: str) -> list[str]: ...
    def family_id(self, market_id: str) -> str: ...
    def duplicate_cluster_id(self, market_id: str) -> str | None: ...
```

This is where the Kalshi–Polymarket linkage lives.

### 4.4 Example builder

```python
# builders.py
import pandas as pd

class ExampleBuilder:
    def __init__(self,
                 store: CanonicalStore,
                 relations: RelationIndex,
                 global_features: GlobalFeatureProvider,
                 cfg: ExperimentConfig):
        self.store = store
        self.relations = relations
        self.global_features = global_features
        self.cfg = cfg

    def build_manifest(self) -> pd.DataFrame:
        """One row per (target, H, Delta) example."""
        ...

    def build_context_rows(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """One row per usable sibling snapshot."""
        ...

    def validate(self, manifest: pd.DataFrame, context_rows: pd.DataFrame) -> None:
        """Hard leakage checks and split-integrity checks."""
        ...
```

This builder should own all cutoff logic. Do not duplicate that logic in training code.

### 4.5 Dataset and collator

```python
# dataset.py
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

@dataclass
class Batch:
    stale: torch.Tensor              # [B, Ds]
    global_feats: torch.Tensor       # [B, Dg]
    context: torch.Tensor            # [B, N, Dc]
    context_mask: torch.BoolTensor   # [B, N]
    relation_type: torch.LongTensor  # [B, N]
    platform_id: torch.LongTensor    # [B, N]

    stale_logit: torch.Tensor        # [B]
    y_delta_logit: torch.Tensor      # [B]
    y_current_logit: torch.Tensor    # [B]
    y_outcome: torch.Tensor          # [B]
    y_reprice_24h: torch.Tensor      # [B]

class BeliefUpdateDataset(Dataset):
    def __init__(self, manifest_df, context_df, split: str):
        ...
    def __len__(self) -> int:
        ...
    def __getitem__(self, idx: int) -> dict:
        ...

def collate_examples(examples: list[dict]) -> Batch:
    """Pad variable-size context sets and build masks."""
    ...
```

### 4.6 Unified model and context adapters

This is the key design improvement. Use one model shell and swap only the context adapter.

```python
# model.py
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelOutput:
    z: torch.Tensor
    delta_logit: torch.Tensor
    current_logit: torch.Tensor
    outcome_logit: torch.Tensor | None = None

class ContextAdapter(nn.Module):
    out_dim: int
    def forward(self,
                stale_repr: torch.Tensor,
                context: torch.Tensor,
                context_mask: torch.Tensor,
                relation_type: torch.Tensor,
                platform_id: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ZeroContextAdapter(ContextAdapter):
    """Stale-only baseline."""
    ...

class AggStatsAdapter(ContextAdapter):
    """Mean/max/count raw-context summary, no learned set encoder."""
    ...

class SetContextAdapter(ContextAdapter):
    """Target-conditioned learned set encoder."""
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int,
                 num_relation_types: int = 2, num_platforms: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        ...
```

```python
class BeliefUpdater(nn.Module):
    def __init__(self,
                 stale_dim: int,
                 global_dim: int,
                 context_adapter: ContextAdapter,
                 hidden_dim: int = 64,
                 aux_outcome: bool = True):
        super().__init__()
        ...

    def encode(self, batch: Batch) -> torch.Tensor:
        ...

    def forward(self, batch: Batch) -> ModelOutput:
        """
        Returns:
            z                compact context representation
            delta_logit      predicted hidden update
            current_logit    stale_logit + delta_logit
            outcome_logit    optional auxiliary head
        """
        ...
```

Concrete instantiations:

* `BeliefUpdater(..., ZeroContextAdapter(...))` → stale-only
* `BeliefUpdater(..., AggStatsAdapter(...))` → raw aggregated context
* `BeliefUpdater(..., SetContextAdapter(..., out_dim=128))` → learned uncompressed
* `BeliefUpdater(..., SetContextAdapter(..., out_dim=32))` → learned compact

### 4.7 Trainer and corrupted-context control

```python
# trainer.py
from typing import Optional

class PrimaryTrainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def fit(self,
            model: BeliefUpdater,
            train_loader,
            val_loader) -> BeliefUpdater:
        ...

    def loss(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        ...

    def predict(self, model: BeliefUpdater, loader) -> "pd.DataFrame":
        ...

def corrupt_context_in_batch(batch: Batch) -> Batch:
    """Permute context tensors and masks across examples while keeping labels fixed."""
    ...
```

The corrupted-context model should reuse the exact same trainer and model. The only difference is that `corrupt_context_in_batch` is applied before the forward pass.

### 4.8 Probe training and evaluation

```python
# probes.py
class ProbeTrainer:
    def fit_outcome_probe(self, z_train, x_train, y_train, z_val, x_val, y_val):
        ...
    def fit_repricing_probe(self, z_train, x_train, y_train, z_val, x_val, y_val):
        ...
    def fit_trust_probe(self, z_train, x_train, y_train, z_val, x_val, y_val):
        ...
```

```python
# eval.py
class Evaluator:
    def primary_metrics(self, pred_df) -> dict[str, float]:
        ...
    def transfer_metrics(self, pred_df) -> dict[str, float]:
        ...
    def hard_slice_metrics(self, pred_df) -> dict[str, dict[str, float]]:
        ...
    def bootstrap_ci(self, pred_df, group_col: str, n_boot: int = 1000):
        ...
```

### 4.9 Minimal experiment runner

```python
# run_experiment.py
def run_experiment(cfg: ExperimentConfig) -> None:
    store = ...
    relations = ...
    global_provider = ...

    builder = ExampleBuilder(store, relations, global_provider, cfg)
    manifest = builder.build_manifest()
    context_rows = builder.build_context_rows(manifest)
    builder.validate(manifest, context_rows)

    train_loader, val_loader, test_loader = ...
    models = {
        "stale_only": ...,
        "raw_agg": ...,
        "set_large": ...,
        "set_compact": ...,
        "set_compact_corrupted": ...,
    }

    for name, model in models.items():
        trainer = PrimaryTrainer(cfg)
        fitted = trainer.fit(model, train_loader, val_loader)
        preds = trainer.predict(fitted, test_loader)
        ...
```

---

## 5. What you should implement first

In practice, the order should be:

1. `CanonicalStore`
2. `RelationBuilder`
3. `ExampleBuilder.validate()`
4. `BeliefUpdateDataset + collate_examples`
5. `BeliefUpdater + ContextAdapter`
6. `PrimaryTrainer`
7. `Evaluator`

That order matters because almost all paper risk is in **data construction and leakage control**, not in the neural code.

The highest-leverage move now is to implement the relation builder, the example builder with hard assertions, and the unified adapter-based model shell first.
