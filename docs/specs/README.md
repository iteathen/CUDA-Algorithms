# Specifications

CUDA-Algorithms production behavior is governed only by **Accepted** bounded specifications.

## Specification maturity

During initial activation, drafts are allowed to change aggressively while implementation and consumer evidence are still exposing the correct abstraction.

```text
Working Draft -> Candidate -> Accepted
```

- **Working Draft:** exploratory; no compatibility promise; exact names, layouts, bounds and decomposition may change or be deleted.
- **Candidate:** intended production shape; reference semantics plus materially different consumer mappings exist; changes remain allowed but must reconcile dependent evidence.
- **Accepted:** production semantic authority; compatibility/evolution rules apply.

Prototype/reference evidence must record the exact revision it tested. Do not add compatibility shims for Working Draft APIs.

## Current authority

- [SPEC-0001 — Native Boundary and JavaScript/Device-JS Implementation](SPEC-0001-native-boundary-and-js-only-implementation.md): **Accepted** implementation and ownership boundary.

## Working activation drafts

- [SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining](SPEC-0002-algorithm-plans-active-extents-and-device-chaining.md): **Working Draft** common plan/resource/device-count chaining semantics.
- [SPEC-0003 — Core Sequence and Keyed GPU Primitives](SPEC-0003-core-sequence-and-keyed-primitives.md): **Working Draft** first reusable primitive spine: scan, reduce, select-indices, gather, stable radix sort, run-length encode and reduce-by-key.
- [SPEC-0004 — Device Worksets and Fixed-Point Closure](SPEC-0004-device-worksets-and-fixed-point-closure.md): **Working Draft** GPU-owned ranked closure and later general monotone workset/fixed-point semantics.

No scan/sort/reduce/closure production API is accepted merely because it appears in the charter, design notes, these Working Drafts, or issue #3.

Each Accepted production profile must define exact semantics, finite limits, resource requirements, determinism/stability behavior, failure truth and qualification evidence through public lower contracts.
