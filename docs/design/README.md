# Design notes

Design notes record active architecture reasoning and may change as implementation evidence arrives. They are not production contract authority unless an accepted ADR/spec explicitly adopts their contents.

Current:

- [First Production Profile Design — GPU-Resident Parallel Algorithms](2026-09-09-first-profile-design.md): working decomposition for plan/device-chaining semantics, sequence/keyed primitives, and GPU-owned workset/closure execution under issue #3.
- [Permutation-First Ordering Refinement](2026-09-09-permutation-first-ordering.md): replaces pair movement as the primary first-profile ordering stud with stable ordering of an index sequence by external key storage.
- [Exact Grouping and Canonicalization Ownership Boundary](2026-09-09-canonicalization-ownership-boundary.md): keeps exact consumer/domain equality with its natural owner while reusing CUDA-Algorithms ordering/scan/select/gather mechanics for segmentation and canonicalization structure.

Working design should be updated or superseded when experiments materially change the architecture; do not preserve stale design text merely as historical compatibility.
