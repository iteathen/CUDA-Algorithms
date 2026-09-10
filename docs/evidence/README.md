# Evidence

This directory records qualification/prototype evidence that materially affects CUDA-Algorithms design and specification decisions.

Evidence is not specification authority. Every record should name the exact revision/environment it exercised and keep implementation/reference/native/performance claims separate.

Current:

- [2026-09-09 core reference qualification](2026-09-09-core-reference-qualification.md): first deterministic JavaScript oracle results for the SPEC-0002/SPEC-0003 Working Draft primitive subset.
- [2026-09-09 first GPU-slice portable CUDA-JS boundary qualification](2026-09-09-first-gpu-slice-portable-boundary.md): exact Device-JS frontend acceptance plus public prepared-DAG mock composition for the device-count-driven stable-selection experiment; not native GPU evidence.
- [2026-09-09 permutation-first ordering portable qualification](2026-09-09-permutation-ordering-portable-boundary.md): 19/19 combined reference tests plus Device-JS frontend and public prepared-DAG mock evidence for stable external-key index ordering; not native GPU evidence.
