# SPEC-0001: Native Boundary and JavaScript/Device-JS Implementation

**Status:** Accepted
**Date:** 2026-09-09

## Outcome

Define the implementation boundary for CUDA-Algorithms before any production algorithm profile is authorized.

CUDA-Algorithms owns provider-neutral reusable algorithm semantics and plans. CUDA-JS remains the sole lower owner of Device-JS translation, compilation/linking, native CUDA/provider access, device/context ownership, memory/view realization, operation scheduling, transfers, publication, failure provenance and cleanup.

## Maintained source rule

Maintained CUDA-Algorithms implementation/reference source is JavaScript/TypeScript plus restricted Device-JS submitted through public CUDA-JS contracts.

The repository must not add or vendor:

- C or C++ implementation source;
- CUDA C++ source or `.cu` implementation files;
- PTX/cubin/SASS as maintained algorithm authority;
- N-API/node-addon/native FFI bindings;
- copied CUDA Driver/Runtime ABI declarations;
- private imports from CUDA-JS internals;
- provider-specific public vocabulary merely to expose an accelerator.

Generated/native artifacts produced privately by public CUDA-JS compiler/provider contracts are not maintained source in this repository and remain owned by their lower lifecycle.

## Semantic/provider separation

An algorithm contract must define its mathematical/ordering semantics independently of realization.

For example, a future radix-sort profile may define:

```text
key domain
value association
stable/unstable policy
ordering relation
active extent
aliasing rules
scratch requirement
bounded failure/backpressure
semantic identity
```

without exposing CUDA block sizes, streams, CUB classes, native pointers or provider handles.

A Device-JS realization, a future qualified library-backed realization, or another provider may implement the same accepted semantic contract. Provider identity may participate in execution/qualification identity where material but does not redefine the public algorithm meaning.

## Generic mechanism routing

If implementing an accepted algorithm demonstrates a missing reusable GPU-language/runtime primitive—such as shared memory, barriers, warp vote/shuffle, local arrays, richer atomics or another consumer-neutral execution mechanism—that gap routes to CUDA-JS.

CUDA-Algorithms must not create a shadow Device-JS dialect or private native escape hatch to bypass a lower-layer gap.

## Resource ownership

CUDA-Algorithms may own logical algorithm plans, scratch-size requirements, logical buffer roles, active extents and algorithm-specific liveness constraints.

CUDA-JS owns actual device allocations/views and native execution lifecycle. CUDA-MM, if activated for a profile, may own generic physical placement/reuse/spill/migration policy from owner-supplied logical constraints. CUDA-Algorithms does not become a physical allocator or storage engine.

## Device-owned progression

A future accepted frontier/workset/fixed-point profile may allow GPU-resident progression without a host advancement loop. The semantic contract may define bounded workset transitions and convergence conditions, while CUDA-JS owns the execution/publication mechanisms used to realize them.

Node/host administration must not be silently promoted into algorithm semantics. A consumer such as BSFP may require that semantic progression remain device-owned while Node only administers opaque epochs/checkpoints; that consumer requirement can motivate a generic profile but does not move BSFP meaning here.

## Bounds and determinism

Every accepted production profile must state finite bounds or explicit resource-pressure/backpressure behavior for all material work and scratch state. Accidental JavaScript `Number`/bitwise limits, implicit unbounded queues, and provider-dependent undefined ordering are forbidden.

Determinism, stability, associativity requirements, floating-point reduction policy and duplicate/key equivalence must be explicit per algorithm family rather than inferred from implementation.

## Qualification rule

Portable/reference tests prove semantic behavior only. Native/provider support and performance require exact compatible-profile evidence through public CUDA-JS paths and independent/reference comparison appropriate to the algorithm.

No repository existence, benchmark prototype, provider availability or first-consumer success is itself production-support authority.

## Non-goals

- defining the first algorithm family;
- selecting CUB/CCCL or another provider;
- authorizing arbitrary CUDA source;
- owning CUDA-JS parallel-language primitives;
- owning Tensor/dataframe/graph/search/domain semantics;
- making a universal hidden scheduler or memory manager.
