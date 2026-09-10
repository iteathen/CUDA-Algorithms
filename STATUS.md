# CUDA-Algorithms Status

**Updated:** 2026-09-09

**Architecture/ownership:** Accepted under ADR-0001 and SPEC-0001.
**First maintained algorithm surface:** Candidate; implemented on `design/first-algorithm-profile`.
**Native CUDA-Algorithms result qualification:** not yet run.
**Performance support/claims:** none.

## Current work

- #1 ownership/bootstrap — completed.
- #2 repository controls — `main` protected; remaining admin parity tracked separately and does not block algorithm work.
- #3 first consumer-backed algorithm profile — active.

Protected `main` currently includes PR #6 read-only CI/document verification. The feature branch merged that protected-main baseline before continued development.

## Candidate specifications

- **SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining:** Candidate.
- **SPEC-0003 — Stable Index Selection and Permutation Ordering:** Candidate.
- **SPEC-0004 — Device Worksets and Fixed-Point Closure:** Working Draft.

Candidate is not Accepted compatibility/support authority. Breaking corrections remain allowed before acceptance when qualification exposes a better complete design.

## Maintained candidate implementation

Package identity in development:

```text
cuda-algorithms@0.1.0-alpha.0
peer: cuda-js@0.1.0-alpha.19
```

Maintained public candidate surface:

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

## Current portable evidence

Exact portable candidate evidence uses:

```text
CUDA-JS: e9837f20acf7901d445a1e7a2045459a1ae0118a
Node:    v26.7.0
```

Latest complete green boundary includes:

- 23/23 independent/reference tests;
- 5/5 maintained candidate API tests;
- Device-JS inspection of correctness kernels;
- public prepared-DAG composition;
- exact-view write-conflict rejection and legal read/read reuse;
- physical qualification harness syntax validation.

Current correctness kernels require only accepted Device-JS facilities (`globalX`, CAS and relaxed device-scope atomic status observation). Shared memory/warp/local-array widening is not required for correctness and remains performance-driven.

## CUDA-JS physical substrate

CUDA-JS current revision `e9837f20acf7901d445a1e7a2045459a1ae0118a` records its Windows gate-32 compatible pair as passed, reviewed and owner-approved on the exact recorded Windows x64 / GTX 1660 Ti / driver 610.74 / CUDA 13.3 / Node 26.7.0 profile.

That establishes a physically available lower substrate for the recorded host profile. It does **not** automatically qualify CUDA-Algorithms numerical/semantic results.

`experiments/native-qualification/run.mjs` now exercises the maintained candidate API, including valid/error stable selection and two-/three-word stable ordering against independent references. The CUDA-Algorithms physical run remains outstanding.

## Alias ownership boundary

A cleanup-safe falsifier proved that current CUDA-JS prepared submission does not enforce an upper algorithm's intra-node same-view non-alias rule. That behavior is compatible with the lower prepared-DAG ownership boundary.

CUDA-Algorithms now rejects exact same-view conflicts whenever either role writes and permits pure read/read reuse.

Different sibling views can still overlap one underlying allocation without enough public information for CUDA-Algorithms to classify them. The consumer-neutral lower relation is tracked by **CUDA-JS #260**. Until it exists or the Candidate is explicitly narrowed, CUDA-Algorithms must not claim full overlapping-sibling-view detection.

Evidence: `docs/evidence/2026-09-09-device-view-alias-boundary.md`.

## Next executable work

1. Build the first **ranked-closure** reference/vertical slice under Working Draft SPEC-0004.
2. Keep derivation/equality/domain semantics with consumers; identify only the generic workset/epoch algebra that survives consumer deletion.
3. Preserve device-resident progression and bounded administrative yields; no Node semantic loop.
4. Run the maintained candidate native harness on the qualified Windows CUDA-JS substrate when that host execution path is available to the acting agent/operator.
5. After native correctness, measure the O(n²) correctness kernels before requesting shared-memory/warp CUDA-JS widening.

## Claim limits

- Portable/mock/frontend evidence is not native CUDA-Algorithms evidence.
- CUDA-JS hardware qualification does not transfer automatically to upper algorithm results.
- Correctness-first kernels make no performance claim.
- Candidate specs are not Accepted compatibility authority.
- No BSFP, dataframe, graph or other consumer semantics belong in CUDA-Algorithms.
