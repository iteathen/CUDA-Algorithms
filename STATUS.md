# CUDA-Algorithms Status

**Updated:** 2026-09-09

**Architecture/ownership:** Accepted under ADR-0001 and SPEC-0001.
**First maintained algorithm surface:** Candidate; integrated to protected `main` through PR #7.
**Current development branch:** `feature/ranked-closure`.
**Native CUDA-Algorithms result qualification:** not yet run.
**Performance support/claims:** none.

## Current work

- #1 ownership/bootstrap — completed.
- #2 repository controls — `main` protected; remaining admin parity tracked separately and does not block algorithm work.
- #3 first consumer-backed algorithm profile — Candidate primitive milestone integrated; ranked closure remains active Working-Draft work.

PR #7 squash-integrated the first maintained Candidate implementation to protected `main` at `6dac37af888c70e437caac3c0eb7224cd87dd73f`. Superseded first-slice/order prototypes were retired before merge; their evidence remains in `docs/evidence/` and PR/Git history. `experiments/native-qualification/` remains active because physical upper-layer qualification is still outstanding.

## Candidate specifications

- **SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining:** Candidate.
- **SPEC-0003 — Stable Index Selection and Permutation Ordering:** Candidate.
- **SPEC-0004 — Device Worksets and Fixed-Point Closure:** Working Draft.

Candidate is not Accepted compatibility/support authority. Breaking corrections remain allowed before acceptance when qualification exposes a better complete design.

## Maintained Candidate implementation

Development package identity:

```text
cuda-algorithms@0.1.0-alpha.0
peer: cuda-js@0.1.0-alpha.20
```

Maintained public Candidate surface:

```text
createStableSelectIndicesU32Plan(...)
createStableLexicographicOrderIndicesU32Plan(...)
```

Both plans:

- use public CUDA-JS only;
- operate over public `u32` device views;
- use device-resident active counts;
- return the ordinary CUDA-JS operation from `submit()`;
- do not call `wait()` or read results inside the production API;
- own bounded plan-resource cleanup without creating a second native-operation lifecycle.

Stable lexicographic ordering keeps consumer records in place and reorders an index sequence by external key-word columns. The current prepared-DAG realization supports at most 31 key words because CUDA-JS currently admits 32 prepared kernel nodes; this is a realization limit, not an algorithmic key-width limit.

## Portable evidence

Exact Candidate evidence uses:

```text
CUDA-JS: 98e2ebc942c14d63acf4dd82e912dd548c363a05
package: cuda-js@0.1.0-alpha.20
Node:    v26.7.0
```

The integrated milestone includes:

- independent/reference semantics;
- maintained Candidate API qualification;
- Device-JS/prepared-DAG composition;
- same-range and partially overlapping sibling-view rejection through CUDA-JS public range truth;
- legal read/read reuse;
- ranked-index-closure reference/shard-invariance tests;
- syntax-valid maintained physical qualification harness;
- permanent read-only CI.

Current correctness kernels require only accepted Device-JS facilities (`globalX`, CAS and relaxed device-scope atomic status observation). Shared memory/warp/local-array widening is not required for correctness and remains performance-driven.

## CUDA-JS physical substrate

CUDA-JS gate-32 physical evidence was recorded on its exact alpha.19 Windows compatible pair. CUDA-JS alpha.20 at `98e2ebc942c14d63acf4dd82e912dd548c363a05` then added the pure public view-range relation.

That establishes the lower runtime/mechanism path and the alias-relation stud required by CUDA-Algorithms. It does **not** automatically qualify CUDA-Algorithms numerical/semantic results.

`experiments/native-qualification/run.mjs` exercises the maintained Candidate API against independent references. The CUDA-Algorithms physical run remains outstanding and must record the exact source/CUDA-JS/Node/GPU/driver/provider tuple actually exercised.

## Ranked closure

The reference model now establishes a consumer-neutral indexed ranked-closure shape:

```text
bounded item universe
+ explicit rank per item
+ finite maxEmissionsPerItem
+ consumer-owned dependency derivation
+ CUDA-Algorithms-owned activation/progression
```

Reference tests cover nested dependency-style and data-lineage-style consumers, shard-size invariance, duplicate activation, rank monotonicity, fanout bounds, target bounds, and rank-zero behavior.

The next implementation target is the first GPU-owned ranked-closure vertical slice. Consumer meaning remains outside CUDA-Algorithms; Node remains administrative and must not perform mathematical frontier progression.

## Branch / PR hygiene

Durable historical review lives in merged PRs, not stale branch refs. After PR #7, the intended live branch set is:

```text
main
feature/ranked-closure
```

The old `bootstrap/open-source-foundation`, `bootstrap/governance-readback`, and `design/first-algorithm-profile` refs contain no required live work once their merged PR history is retained and are safe to retire.

## Claim limits

- Portable/mock/frontend evidence is not native CUDA-Algorithms evidence.
- CUDA-JS hardware qualification does not transfer automatically to upper algorithm results.
- Correctness-first kernels make no performance claim.
- Candidate specs are not Accepted compatibility authority.
- No BSFP, dataframe, graph or other consumer semantics belong in CUDA-Algorithms.
