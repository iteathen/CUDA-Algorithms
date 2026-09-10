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

Protected `main` includes PR #6 read-only CI/document verification. The feature branch merged that protected-main baseline before continued development.

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

## Current portable evidence

Exact portable Candidate evidence uses:

```text
CUDA-JS: 98e2ebc942c14d63acf4dd82e912dd548c363a05
package: cuda-js@0.1.0-alpha.20
Node:    v26.7.0
```

Latest complete green boundary includes:

- 23/23 independent/reference tests;
- 6/6 maintained Candidate API tests;
- Device-JS inspection of correctness kernels;
- public prepared-DAG composition;
- same-range write-conflict rejection;
- partially overlapping sibling-view rejection through public CUDA-JS range truth;
- legal read/read reuse;
- physical qualification harness syntax validation.

Current correctness kernels require only accepted Device-JS facilities (`globalX`, CAS and relaxed device-scope atomic status observation). Shared memory/warp/local-array widening is not required for correctness and remains performance-driven.

## CUDA-JS physical substrate

CUDA-JS gate-32 physical evidence was recorded on the exact alpha.19 Windows x64 / GTX 1660 Ti / driver 610.74 / CUDA 13.3 / Node 26.7.0 compatible pair and remains valid for that recorded pair.

CUDA-JS main then added the pure public view-range relation in alpha.20 at `98e2ebc942c14d63acf4dd82e912dd548c363a05`. The relation itself performs no actor/native work and keeps allocation/native identity private.

That gives CUDA-Algorithms the lower relation required for full write-range alias admission policy. It does **not** automatically qualify CUDA-Algorithms numerical/semantic results or silently transfer the earlier package-level native claim to alpha.20.

`experiments/native-qualification/run.mjs` exercises the maintained Candidate API, including valid/error stable selection and two-/three-word stable ordering against independent references. The CUDA-Algorithms physical run remains outstanding and must record the exact alpha.20 revision actually exercised.

## Alias ownership boundary — resolved at public-contract level

Historical falsification showed that prepared-DAG hazards do not own an upper algorithm's intra-node alias policy. CUDA-JS #260 therefore added the correct lower LEGO stud:

```text
inspectDeviceViewRelation(a, b)
  -> same-range | overlap | disjoint
```

CUDA-Algorithms now owns the policy over that lower fact:

- read/read ranges may overlap where the algorithm permits reuse;
- any pair involving a write must be `disjoint`;
- `same-range` or `overlap` rejects before prepared algorithm submission.

This now covers distinct sibling views without exposing their parent allocation identity.

Evidence: `docs/evidence/2026-09-09-device-view-alias-boundary.md`.

## Next executable work

1. Build the first **ranked-closure** reference/vertical slice under Working Draft SPEC-0004.
2. Keep derivation/equality/domain semantics with consumers; identify only the generic workset/epoch algebra that survives consumer deletion.
3. Preserve device-resident progression and bounded administrative yields; no Node semantic loop.
4. Run the maintained Candidate native harness on the available Windows CUDA-JS host when direct host execution is available to the acting agent/operator.
5. After native correctness, measure the O(n²) correctness kernels before requesting shared-memory/warp CUDA-JS widening.

## Claim limits

- Portable/mock/frontend evidence is not native CUDA-Algorithms evidence.
- CUDA-JS hardware qualification does not transfer automatically to upper algorithm results or later package revisions.
- Correctness-first kernels make no performance claim.
- Candidate specs are not Accepted compatibility authority.
- No BSFP, dataframe, graph or other consumer semantics belong in CUDA-Algorithms.
