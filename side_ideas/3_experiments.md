# Side Idea 3 Experiments

## Notation

- `A`: the target market whose final outcome we want to predict
- `B`: related market context
  - this may include one related market or a set of related markets
  - examples: same event family, semantically related contracts, logically linked contracts
- `E`: external signals
  - examples: BTC, ETH, macro series, other non-Polymarket covariates

- `raw(A)`: raw local information for the target market `A`
  - for example: recent price path, activity, volatility, liquidity-like features, metadata
- `raw(B)`: raw related-market context
- `raw(E)`: raw external features

- `encoded(B)`: a compact latent representation of related-market context `B`
- `encoded(B,E)`: a compact latent representation of the full non-local context, combining related markets `B` and external signals `E`

Important design choice for the first version:

- keep `A` in raw form;
- use the encoder mainly to compress everything around `A`, not `A` itself.

So the key representation-learning setup is:

- `raw(A) + encoded(B,E)`

---

## Core Experiment Ladder

**predict outcome(A) from raw(A)**  
Local-only baseline. Measures how much signal is already present in the target market alone.

**predict outcome(A) from raw(A) + raw(B)**  
Adds raw related-market context. Tests whether cross-market information helps beyond the local state of `A`.

**predict outcome(A) from raw(A) + encoded(B)**  
Replaces raw related-market context with a compact latent representation. Tests whether related-market context can be compressed without major loss.

**predict outcome(A) from raw(A) + raw(E)**  
Adds raw external signals. Tests whether external markets or macro variables help predict `A`.

**predict outcome(A) from raw(A) + raw(B) + raw(E)**  
Full raw-context upper bound. Measures the best performance available before context compression.

**predict outcome(A) from raw(A) + encoded(B,E)**  
Key experiment for Side Idea 3. Replaces the full non-local world around `A` with a compact context state. Tests whether the rest of the market system can be summarized effectively.

---

## Representation-Focused Variants

**predict outcome(A) from raw(A) + encoded(B) with a frozen encoder**  
Train the encoder first, then freeze it and train only a simple downstream head. Tests whether the representation itself is useful.

**predict outcome(A) from raw(A) + encoded(B,E) with a frozen encoder**  
Stronger version of the frozen-probe test. Checks whether compact non-local context remains predictive without end-to-end adaptation.

**predict outcome(A) from raw(A) + encoded(B,E) with different latent sizes**  
For example `z = 32, 64, 128, 256`. Tests how compact the context representation can be before downstream quality drops.

---

## Hard-Slice Variants

**predict outcome(A) from raw(A) on hard slices**  
Run the local-only baseline on difficult subsets such as uncertain states, thin markets, or high-coherence-gap snapshots.

**predict outcome(A) from raw(A) + raw(B) on hard slices**  
Tests whether raw related-market context helps especially in difficult regimes.

**predict outcome(A) from raw(A) + encoded(B,E) on hard slices**  
Tests whether compact context helps specifically where the target market `A` is incomplete, uncertain, or structurally stressed.

---

## Transfer And Generalization Variants

**predict outcome(A) from raw(A) + encoded(B,E) with low-label adaptation**  
Train the encoder broadly, then fit downstream heads with very few labels. Tests sample efficiency.

**predict outcome(A) from raw(A) + encoded(B,E) on unseen event families**  
Train on one set of event families and test on new ones. Tests whether the representation generalizes beyond seen templates.

**predict outcome(A) from raw(A) + encoded(B,E) on unseen domains**  
Train on one domain and test on another. Tests whether the representation transfers across domains.

**predict outcome(A) from raw(A) + encoded(B,E) for cold-start markets**  
Use a new market `A` with short local history. Tests whether compact context helps when the target market itself has limited data.

---

## Interpretation

The most important comparison is not just:

- `raw(A)` versus `raw(A) + raw(B) + raw(E)`

but:

- `raw(A) + raw(B) + raw(E)` versus `raw(A) + encoded(B,E)`

That comparison answers the core question of Side Idea 3:

- can the non-local market universe around a target market be replaced by a compact latent context state without losing too much task-relevant information?
