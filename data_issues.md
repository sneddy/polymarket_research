# Data Issues

## Resolved-status inconsistency

Current `active`, `closed`, and `final_outcome` fields from the source are not reliably consistent with each other.

Observed failure mode:
- markets can have `active = 1`
- markets can have `closed = 1`
- markets can have non-null `final_outcome`
- while `end_date` is still in the future relative to extraction time

Example implication:
- a market may look "resolved-like" under the current heuristic
- but should not be treated as truly resolved for research evaluation

Why this matters:
- raw `final_outcome` cannot currently be trusted as a clean resolved label
- any benchmark or panel that assumes `final_outcome != null` means truly resolved is vulnerable to leakage / contamination
- historical plots based on market lifetime can accidentally include future-dated markets that were incorrectly admitted into the resolved pool

Current mitigation:
- raw layer now includes a heuristic `resolved` flag based on:
  - `end_date <= synced_at_utc`
- canonical layer supports `resolved_only=True`

Current limitation:
- this heuristic is weaker than a true resolved-state guarantee
- it avoids the obviously wrong case where `end_date` is still in the future
- but it still depends on sync timing rather than an explicit source-side resolution flag

Recommended next step:
- redesign the resolved-market admission rule in the export pipeline
- use a stricter and more explicit resolution criterion than `closed == true` plus extreme outcome prices
