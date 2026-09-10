# Permutation-First Ordering — Portable CUDA-JS Boundary Qualification

**Date:** 2026-09-09
**CUDA-Algorithms branch:** `design/first-algorithm-profile`
**Qualified experiment head:** `3df2e3615a0272cb32683403a7915b58e455b0fe`
**Pinned CUDA-JS authority:** `97c0295ab79add204d4d8ced080a4da4b66149cf`
**Node:** v26.7.0
**GitHub Actions run:** `34420180872`
**Job:** `102693670376`
**Conclusion:** success

## Scope

This record qualifies the permutation-first ordering Working Draft against deterministic reference semantics, the current CUDA-JS Device-JS frontend, and the public CUDA-JS portable prepared-DAG orchestration surface.

It is **not native GPU numerical or performance evidence**.

## Reference result

The combined reference run passed **19/19 tests**:

- 15 existing scan/select/gather/keyed/width/device-extent tests;
- stable ordering relative to a non-identity incoming index sequence;
- permutation-first multiword ordering versus the earlier pair-sort composition and independent tuple oracle;
- 250 deterministic duplicate-heavy multiword fixtures;
- invalid indirect-index rejection before key lookup.

The 250-fixture ordering set covered 1–8 key words and 1–96 records per fixture.

## Device-JS frontend result

CUDA-JS accepted the ordering source under:

```text
SPEC-0013-v1+SPEC-0022-atomic-observation-v1+SPEC-0022-device-publication-v1+SPEC-0014-publication-mailbox-v1
```

Normalized ordering-program identity:

```text
05470a27090aa771134c3e56b6d4546d1eeb148dd93ad97ef9a2804e3217dcf1
```

Public helper usage:

```text
resetOrderingStatus:
  gpu.thread.globalX

stableOrderIndicesByKeyU32:
  gpu.atomic.cas
  gpu.thread.globalX
```

No shared-memory, local-array, warp, cooperative-grid, raw CUDA or private-provider mechanism is required to express the correctness-first ordering semantic.

## Prepared multiword ordering composition

The public CUDA-JS portable mock accepted one prepared 3-node / 2-edge DAG:

```text
reset status
  -> order low word:  indicesA -> indicesB
  -> order high word: indicesB -> indicesA
```

Bindings:

```text
activeCount
highKeys
indicesA
indicesB
lowKeys
status
```

The single prepared operation completed in the mock and lifecycle cleanup was graceful.

## Consequence for Working Draft SPEC-0003

The evidence supports the revised primary ordering stud:

```text
stableOrderIndicesByKey(externalKeys, indicesIn) -> indicesOut
```

rather than requiring direct key/pair movement as the public abstraction.

The provider remains free to gather keys internally and delegate to a radix-sort implementation. Therefore this result supports the semantic boundary without constraining the optimized realization to O(N²) indirect ranking.

## What remains unproved

- physical GPU output correctness;
- performance of indirect versus internally materialized/provider radix realization;
- `u64` device profile;
- selected-bit-range device realization;
- production workspace formulas;
- wide-key physical scaling;
- downstream grouping/canonicalization shape;
- RankedClosure.

## Next design falsifier

Test whether grouping/canonicalization should follow the same permutation-first principle: mark run/group boundaries by external keys over an already ordered index sequence, then compose with scan/select rather than eagerly gathering key or record payloads.

This is especially relevant to BSFP exact canonicalization and CUDA-DATA grouping. If the abstraction survives those materially different consumers and reference tests, revise the Working Draft before implementing direct RLE as a primary public stud.
