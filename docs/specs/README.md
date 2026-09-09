# Specifications

CUDA-Algorithms production behavior must be governed by accepted bounded specifications before implementation.

Current foundation:

- [SPEC-0001 — Native Boundary and JavaScript/Device-JS Implementation](SPEC-0001-native-boundary-and-js-only-implementation.md): accepted implementation and ownership boundary.

No scan/sort/reduce/closure production API is accepted merely because it appears in the charter or issue #3. Each production profile must define exact semantics, finite limits, resource requirements, determinism/stability behavior, failure truth and qualification evidence.
