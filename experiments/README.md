# Experiments

Only active qualification experiments should remain here.

## Active

- `native-qualification/` — physical CUDA qualification for the maintained Candidate API. This remains qualification-only code and may wait/read back results because it compares GPU outputs against independent references.

## Retired prototypes

The initial `first-gpu-slice/` and `stable-permutation-ordering/` prototypes were intentionally removed after their semantics moved into the maintained `src/` implementation.

Their durable conclusions are preserved in:

- `docs/evidence/2026-09-09-first-gpu-slice-portable-boundary.md`
- `docs/evidence/2026-09-09-permutation-ordering-portable-boundary.md`
- `docs/evidence/2026-09-09-device-view-alias-boundary.md`

The evidence records name exact historical commits and workflow runs. Git/PR history preserves the prototype source itself; keeping superseded duplicate implementation trees on `main` would create multiple apparent authorities and violate the project’s LEGO/ownership discipline.
