# Specifications

CUDA-Algorithms production semantic authority exists only in **Accepted** bounded specifications. Candidate specifications authorize maintained qualification work but remain breakable before acceptance when evidence requires correction.

## Specification maturity

```text
Working Draft -> Candidate -> Accepted
```

- **Working Draft:** exploratory; no compatibility promise; exact names, layouts, bounds and decomposition may change or be deleted.
- **Candidate:** maintained intended shape with reference/implementation/consumer evidence; breaking corrections remain allowed before acceptance.
- **Accepted:** production semantic authority; compatibility/evolution rules apply.

## Current authority

- [SPEC-0001 — Native Boundary and JavaScript/Device-JS Implementation](SPEC-0001-native-boundary-and-js-only-implementation.md): **Accepted** ownership/native boundary.

## Candidate activation contracts

- [SPEC-0005 — Checked u32 scan and segment offsets](SPEC-0005-checked-scan-and-segment-offsets.md): **Candidate** under explicit owner implementation request #9; checked addition, head-flag segmentation, owned finite workspace and device status/count composition.

- [SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining](SPEC-0002-algorithm-plans-active-extents-and-device-chaining.md): **Candidate** common nonblocking plan, device-resident active-count, semantic-status, resource and alias-ownership contract.
- [SPEC-0003 — Stable Index Selection and Permutation Ordering](SPEC-0003-core-sequence-and-keyed-primitives.md): **Candidate** first maintained `u32` family: stable select-indices plus stable lexicographic ordering of an index sequence by external key-word columns.

Neither Candidate is yet native CUDA-Algorithms support authority. Physical result qualification and the distinct-sibling-view alias boundary remain acceptance gates.

## Working activation draft

- [SPEC-0004 — Device Worksets and Fixed-Point Closure](SPEC-0004-device-worksets-and-fixed-point-closure.md): **Working Draft** GPU-owned ranked closure and later general monotone workset/fixed-point semantics. Its callback/composition and epoch boundary remain deliberately open until a real vertical slice settles them.

## Deferred families

The earlier broad primitive draft mentioned general scan/reduce, gather, RLE, reduce-by-key and other operations. Those remain plausible CUDA-Algorithms territory but are **not Candidate APIs** merely because reference helpers or design notes exist.

Every later family must earn its own bounded semantics, consumer-neutral ownership, reference behavior, resource/failure contract and qualification evidence.
