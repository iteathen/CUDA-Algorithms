# SPEC-0002: Algorithm Plans, Active Extents, and Device Chaining

**Status:** Working Draft
**Date:** 2026-09-09
**Issue:** #3

> This document is intentionally mutable during the first implementation cycle. It is not accepted production authority and carries no compatibility promise until promoted to Candidate and then Accepted.

## Outcome

Define the common provider-neutral execution contract shared by CUDA-Algorithms families without creating a second CUDA runtime, memory model, operation lifecycle, tensor model, or consumer scheduler.

The central design requirement is that one GPU-produced logical extent/result may feed later GPU work without a mandatory host readback or host semantic advancement step.

## Ownership

CUDA-Algorithms owns:

- immutable logical algorithm-plan semantics;
- input/output roles and active-extent meaning;
- algorithm-specific workspace requirements;
- stability/determinism policy;
- device-chainable result/count semantics;
- algorithm-level resource-pressure and semantic-status meaning.

CUDA-JS owns:

- device allocations and typed views;
- Device-JS compilation/linking;
- native providers and CUDA mechanisms;
- stream/event/operation scheduling;
- transfer mechanics;
- publication mailboxes;
- native failure provenance and cleanup.

CUDA-Algorithms must not expose raw native addresses, CUDA streams/events, provider objects, CUB types, PTX, CUDA C++ or private CUDA-JS state.

## Data binding

The first profile operates on public CUDA-JS device allocations/views.

Algorithm plans describe logical roles such as:

```text
input
output
key
index
flag
count
workspace
status/control
```

These are algorithm roles, not new memory-capability types.

A plan must not infer tensor shape, table schema, graph meaning, proof-record meaning, or consumer ownership from a view.

## Active extent

Every sequence-like operation has:

```text
capacity       maximum item count addressable by the bound buffers
active extent number of logically active items for this invocation
```

The active extent may be represented in one of two forms.

### Host-fixed extent

A host-known non-negative safe integer validated before submission.

Requirements:

- `active <= capacity`;
- all bound views cover the required active range;
- all arithmetic used to derive byte ranges/workspace is checked;
- invalid requests reject before native work.

### Device-resident extent

A one-element CUDA-JS device view containing an unsigned `u32` or `u64` count plus a host-known capacity bound.

Requirements:

- later GPU stages consume the count directly on device;
- Node is not required to read the count between stages;
- the plan declares the count width and maximum capacity;
- every implementation guards `count <= capacity` before touching items beyond the bound;
- over-capacity state must become an explicit semantic failure/status and must never silently truncate or wrap;
- a downstream stage in the same device chain must observe upstream failure/status and avoid manufacturing a valid result from invalid inputs.

The implementation shape of the device status/control record is deliberately not frozen in this Working Draft.

## Count widths

`u32` and `u64` are both candidate count/index widths.

A plan may select `u32` where its explicit capacity proves the value fits. Large logical problems may be sharded into bounded batches without forcing every local index to `u64`.

No public contract may imply that `u32` local counts limit the total size of an out-of-core or multi-shard problem.

## Device chaining

A device-chainable operation writes its produced count and semantic status to device-resident state that later operations can consume without host interpretation.

Conceptually:

```text
producer
  -> device count/status
  -> algorithm A
  -> device count/status
  -> algorithm B
```

The host may submit/prepare the operations and may asynchronously observe administrative progress, but correctness must not require a host data-dependent loop over item records or produced counts.

## Host nonblocking rule

CUDA-Algorithms production execution must not synchronously block the Node event loop waiting for GPU completion.

Host-side JavaScript may perform bounded administration such as:

- plan normalization;
- capability validation;
- allocation request orchestration;
- asynchronous submission;
- registration of completion/status observation;
- checkpoint or persistence administration.

It must not perform the mathematical content of a GPU-owned algorithm while pretending that work is GPU-resident.

## Algorithm semantic status

The common model must distinguish native operation completion from algorithm semantic validity.

Candidate semantic states include:

```text
ok
invalid-extent
overflow
capacity-exhausted
budget-yield
converged
administrative-yield
```

Not every primitive admits every state. Exact spelling and the physical device representation remain open in this Working Draft.

A CUDA operation can complete successfully at the runtime level while its algorithm result reports a bounded semantic condition such as `capacity-exhausted`; the library must not collapse these categories.

## Workspace

Every plan states its logical workspace requirement as a bounded byte/alignment record derived from material inputs such as:

```text
algorithm family/version
input/output dtype
capacity
selected stability/determinism policy
provider-independent algorithm variant where semantic
```

The first expert profile should prefer explicit workspace binding/reuse over hidden unbounded allocation.

CUDA-Algorithms may request ordinary CUDA-JS memory for convenience, but CUDA-JS retains physical allocation/lifecycle ownership. A future CUDA-MM profile may optimize placement/reuse from the logical requirements without changing algorithm meaning.

## Aliasing and access

Each operation must declare exact logical access roles and accepted aliasing.

Default rule: overlapping input/output ranges reject unless the specific algorithm contract explicitly defines in-place semantics.

Provider behavior cannot silently widen or narrow the public aliasing contract.

## Determinism and stability

Every plan records the determinism/stability requirements material to the result.

Candidate dimensions include:

```text
stable-order-required
stable-order-not-required
exact-integer-deterministic
run-to-run-deterministic
cross-device-deterministic
provider-permitted-nondeterministic
```

The first profile should avoid floating-point reduction promises until the exact grouping/reproducibility contract is justified.

## Plan identity

A normalized algorithm plan has deterministic semantic identity over every material public fact, including at least:

```text
contract/family version
operation kind
input/output role schema
dtypes
count/index width
capacity and active-extent form
ordering direction/bit range where applicable
stability/determinism policy
aliasing contract
workspace contract
semantic-status contract
```

Native handles, addresses, streams, generated CUDA source and provider-private tuning do not enter provider-neutral semantic identity.

A separately materialized execution/provider identity may include lower compatible-profile facts where required for cache/evidence truth.

## Execution/lifecycle composition

CUDA-Algorithms must reuse CUDA-JS operation ownership. It must not create a competing definition of submitted/pending/completed/failed native work.

If an algorithm requires multiple kernel/transfer/library operations, its plan may compose them through accepted CUDA-JS dependency/prepared-execution mechanisms. Any higher convenience object represents algorithm planning/result metadata only and must not become a second native-operation lifecycle authority.

## Reference semantics

Each accepted algorithm family requires a deterministic JavaScript/TypeScript reference implementation or another independently understandable oracle for its semantic claims.

Reference execution is qualification evidence and development support. It is not a required CPU production fallback for GPU-owned consumers.

## Mutable-draft rule

While this specification is `Working Draft`:

- exact class/function names are non-authoritative;
- status-record physical layout is open;
- count/control representation may change;
- workspace shapes may change;
- the spec may be split if implementation demonstrates a real ownership seam;
- dependent prototype evidence must record the exact draft revision it tested.

Promotion to Candidate requires at least one real primitive implementation/prototype and mappings from at least two materially different consumers.

Promotion to Accepted requires exact reference conformance, bounded failure/resource behavior, public CUDA-JS composition evidence, and review of the resulting stable public surface.

## Falsifiers

Rework this contract if:

- device-resident active extents cannot be validated without host semantic participation;
- algorithm status requires a second CUDA/native lifecycle abstraction;
- common plan fields become mostly family-specific ceremony;
- arbitrary consumer metadata begins leaking into the common plan;
- workspace ownership cannot remain cleanly separated from physical memory management;
- the nonblocking/device-chaining contract cannot be realized through public CUDA-JS mechanisms.
