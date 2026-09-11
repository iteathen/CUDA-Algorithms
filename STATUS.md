# CUDA-Algorithms Status

## Issue #9 implementation

Branch `codex/checked-scan-issue-9` adds public checked exclusive scan and
segment-offset plans under Candidate SPEC-0005. Finite workspace is plan-owned;
inputs/outputs remain caller-owned. Explicit upstream status and CUDA-JS `after`
support device count/status chaining. Public binding, aliasing, lifetime and
injected construction/cleanup checks pass. Native source f82997ae passes 72
segment fixtures, 59 scan/chaining controls and eight selection comparison passes
on GTX 1660 Ti / CUDA-JS 98e2ebc / Node 26.7.0. Both standalone scan and segment
plans are checked through 262,144 entries. All 49 local tests and CI 34567500406
pass. Exact evidence: `docs/evidence/2026-09-11-checked-scan-api-native.json`.
The 8,192-entry comparison has noisy timings and no clear median speed advantage;
new segment work additionally produces IDs, compact lengths and checked offsets.
Implementation is ready for review; Candidate status is not promoted to Accepted.
Existing selection/ordering semantics and lower pins are unchanged.

## Isolated segment-scan research branch

`codex/oqs-segment-scan` adds an experiment under `experiments/segment-scan/`,
from exact dependency 48ee0ae. It tests parallel prefix computation, dense IDs,
representative lengths and offsets against OQS and nullable-row fixtures.
Portable composition, 44 local tests and 72 native fixture checks pass. Native
source 7d923eb5, recorded in `docs/evidence/2026-09-11-segment-scan-native.json`,
handles 262,144 entries with 11,603,100 device bytes and 0.9921 ms median
submit/wait (three measured repetitions). This is bounded experimental evidence.
The supported-library gap and acceptance work are filed in
[issue #9](https://github.com/iteathen/CUDA-Algorithms/issues/9).
No public export or Candidate specification changed. The inherited ranked
activation checkpoint below remains separate; Connect4's dependency is unchanged.

## Inherited ranked-activation checkpoint

**Updated:** 2026-09-09

**Architecture/ownership:** Accepted under ADR-0001 and SPEC-0001.  
**Protected Candidate milestone:** `main@6dac37af888c70e437caac3c0eb7224cd87dd73f` through PR #7.  
**Current development branch:** `feature/ranked-closure`.  
**Current draft PR:** #8 — generic ranked derived activation vertical slice.  
**Native ranked-activation numerical qualification:** not yet run.  
**Performance support/claims:** none.

## Current work

SPEC-0004 remains Working Draft. The first implemented consumer-composition subprofile is now a deliberately narrower **implicit ranked derived activation** plan rather than a claim that general RankedClosure is complete.

The working public branch surface adds:

```text
createRankedDerivedActivationU32Plan(...)
```

Its exact first-profile consumer seam is a CUDA-JS SPEC-0028 typed device-library export:

```text
(u32 sourceIndex, u32 emissionLane) -> u32 targetIndex | 0xffffffff
```

CUDA-Algorithms owns only the bounded emission lanes, strict rank-descent validation, duplicate-idempotent target activation, deterministic ascending-index compaction, device-resident next extent/status, capacity truth, and prepared-epoch lifecycle.

Consumers retain item/domain meaning, derivation meaning, proof/value/equality/dominance semantics, and rank meaning subject to the strict ordering precondition.

## Scope / universality boundary

This implemented subprofile is appropriate for finite **implicit/static ranked state spaces** whose target dependency is computable from source index plus a finite emission lane.

Materially different examples already exercised through the same typed Device-JS composition path are:

- implicit dependency-DAG activation;
- staged data-lineage dependency activation.

The reference layer additionally covers shard-size invariance, duplicate activation, rank and target bounds, capacity yield, and multiple implicit dependency shapes.

This does **not** claim a universal callback for arbitrary runtime graph/record data. If a later real consumer requires device-resident record/context access, widen the generic seam only from that evidence. Do not invent a generic record ABI preemptively.

## Portable qualification

Exact lower pair:

```text
CUDA-JS: 98e2ebc942c14d63acf4dd82e912dd548c363a05
package: cuda-js@0.1.0-alpha.20
Node:    v26.7.0
```

CUDA-Algorithms PR #8 CI passes on the current branch lineage, including:

- reference ranked-derived-activation semantics;
- implicit-DAG typed Device-JS composition;
- unrelated staged-data-lineage typed Device-JS composition;
- exact signature rejection;
- existing stable-select/stable-order Candidate API checks;
- document validation;
- syntax checks for both physical native qualification harnesses.

Latest recorded successful CI after the two-consumer/native-harness additions:

```text
workflow run: 34438628848
result:       success
```

Connect4 CUDA-BSFP independently consumed the exact ranked-activation implementation through public package surfaces and passed its pinned portable composition workflow. That consumer evidence does not transfer Connect4 proof semantics into CUDA-Algorithms.

## Physical qualification path

`experiments/native-qualification/run-ranked-derived-activation.mjs` is now the dedicated physical harness for this subprofile.

On a real NVIDIA CUDA host it will:

- compile and run the implicit-DAG consumer;
- compile and run the staged-data-lineage consumer;
- compare activated target sets with the independent JavaScript reference;
- verify duplicate idempotence;
- verify output-capacity exhaustion reports the exact required count rather than truncating successfully;
- verify a strict-rank violation produces semantic failure;
- close plan/operation/device resources through public CUDA-JS lifecycle.

Physical NVIDIA execution remains outstanding. Portable/mock/compiler evidence is not native numerical evidence.

## Existing Candidate specifications

- **SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining:** Candidate.
- **SPEC-0003 — Stable Index Selection and Permutation Ordering:** Candidate.
- **SPEC-0004 — Device Worksets and Fixed-Point Closure:** Working Draft.

The ranked-derived-activation subprofile remains Working while SPEC-0004 is Working. It is not Accepted compatibility authority.

## Existing maintained Candidate implementation

Protected `main` continues to own the already-integrated Candidate primitives:

```text
createStableSelectIndicesU32Plan(...)
createStableLexicographicOrderIndicesU32Plan(...)
```

The ranked-derived-activation work is isolated on `feature/ranked-closure`/draft PR #8 and has not changed protected `main`.

## CUDA-JS boundary

No CUDA-JS mutation was required for this slice.

The implementation uses existing public CUDA-JS capabilities only:

- typed Device-JS library composition;
- Device-JS program compilation/linking;
- public device views and opaque range-relation truth;
- prepared operation DAGs;
- device-scope CAS/status observation;
- ordinary CUDA-JS operations and resource lifecycle.

No Python, direct CUDA FFI, C/C++/CUDA C++, hand PTX, private CUDA-JS imports, dynamic device function pointers, or provider-native passthrough were added.

## Next gate

1. Run `node experiments/native-qualification/run-ranked-derived-activation.mjs` on the available real CUDA host and preserve the exact CUDA-Algorithms/CUDA-JS/Node/GPU/driver/provider tuple and raw result.
2. If physical numerical parity passes, reconcile SPEC-0004 wording with the demonstrated **implicit ranked derived activation** subprofile without over-promoting general RankedClosure.
3. Keep broader callback/context shapes open until another real consumer demonstrates a reusable need.
4. Measure performance only after native correctness; route any demonstrated generic lower mechanism gap to CUDA-JS rather than pre-authorizing widening.

## Claim limits

- No native ranked-derived-activation numerical result yet.
- No performance result or recommendation yet.
- SPEC-0004 is still Working Draft.
- The implemented index+lane derivation seam is intentionally narrower than arbitrary consumer callbacks.
- No BSFP, Connect Four, dataframe, graph, proof, W/D/L, or other consumer semantics belong in CUDA-Algorithms.
