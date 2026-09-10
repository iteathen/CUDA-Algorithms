# SPEC-0002: Algorithm Plans, Active Extents, and Device Chaining

**Status:** Candidate
**Date:** 2026-09-09
**Issue:** #3

> Candidate means this contract is stable enough for maintained implementation and native qualification, but it is not yet accepted compatibility authority. Breaking corrections remain allowed before acceptance when evidence exposes a better design.

## Outcome

Define the common provider-neutral execution contract for CUDA-Algorithms plans without creating a second CUDA runtime, memory model, operation lifecycle, tensor model, record model, or consumer scheduler.

The key requirement is that GPU-produced counts/status can feed later GPU work without a mandatory host readback or host semantic advancement step.

## LEGO ownership

CUDA-Algorithms owns:

- algorithm-plan semantics and algorithm-family identity;
- logical input/output/workspace/control roles;
- host-known capacities and device-resident active-extent meaning;
- algorithm semantic status and deterministic/stability requirements;
- algorithm-specific workspace and realization bounds;
- algorithm-specific alias policy.

CUDA-JS owns:

- allocations, device views and their private parent/range truth;
- the consumer-neutral public byte-range relation between live same-runtime views;
- Device-JS parsing/lowering/compilation/linking;
- modules/functions, prepared execution and operation lifecycle;
- native providers, streams/events, transfers, memory ordering and cleanup.

A CUDA-Algorithms plan may compose CUDA-JS capabilities but must not expose or recreate their private/native authority.

## Candidate plan shape

The maintained first profile uses a JavaScript object conceptually equivalent to:

```text
CudaAlgorithmsPlan
  kind
  contract / family
  finite capacities and dtype facts
  workspace / realization metadata
  semantic status vocabulary
  submit(bindings) -> Promise<CudaOperation>
  close() -> Promise<void>
```

`submit()` returns the ordinary CUDA-JS operation. CUDA-Algorithms does not wrap that operation in a competing pending/completed/failed lifecycle.

Plan creation may asynchronously compile/load/prepare bounded lower resources. Plan close owns release of resources created for that plan and must preserve lower cleanup failure truth.

## Device-resident active extent

The first maintained candidate uses:

```text
activeCount: one-element u32 CUDA-JS device view
capacity:    positive host-known safe integer
```

Requirements:

- kernels consume `activeCount` directly on device;
- Node does not need to read the count between algorithm stages;
- every stage proves/guards `activeCount <= capacity` before dereferencing outside the declared range;
- invalid extent becomes algorithm semantic failure, not wrap/truncation;
- zero active items are semantically valid even though physical buffers/launch geometry remain positively bounded;
- a local `u32` count does not impose a `u32` limit on an out-of-core or multi-shard logical problem.

A later `u64` active-count profile requires its own maintained implementation/evidence; it is not implied by the reference oracle alone.

## Node nonblocking / GPU-owned mathematics

Production plan methods must not synchronously block the Node event loop waiting for GPU completion and must not perform the mathematical content of the advertised GPU algorithm on CPU.

Node may:

- validate/normalize finite plan facts;
- create/bind lower capabilities;
- submit operations;
- register asynchronous observation;
- administer persistence, checkpoint and bounded epoch transitions.

Node may not read item records/counts merely to decide mathematical survivor sets, ordering, grouping, predecessor choices or fixed-point progression for a profile advertised as GPU-owned.

Qualification harnesses may deliberately call `wait()` and perform D2H reads to compare GPU results against independent oracles. That is evidence code, not production execution semantics.

## Runtime completion versus algorithm validity

CUDA operation terminality and algorithm semantic validity are separate facts.

A lower operation may complete successfully while the algorithm status reports, for example:

```text
invalid-extent
capacity-exhausted
invalid-input
```

No family may manufacture a valid success payload from an error state unless that exact partial-result meaning is specified.

For capacity failure, a family may preserve an independently valid required count for administration while declaring payload output invalid; silent truncation is forbidden.

## Status/control access discipline

When multiple GPU threads can observe/update the same status/control location in one kernel:

- use one compatible accepted CUDA-JS atomic access discipline for the concurrent region;
- do not mix ordinary concurrent access with atomic access to that same location;
- a separately ordered single-thread initialization kernel may initialize before the concurrent region;
- first-error publication may use compare-and-swap from `ok` to preserve the first observed semantic failure;
- stronger acquire/release/system semantics are required only when the actual cross-location/publication contract needs them.

The current candidate implementations use CUDA-JS device-scope relaxed atomic observation plus CAS for same-location algorithm status. CUDA-JS remains authoritative for those memory-model semantics.

## Data binding and access

Candidate plans bind public one-dimensional CUDA-JS device views and must validate the public facts they rely on, including dtype, minimum element capacity and access authority.

Whole-plan access truth must be stated honestly. A workspace written in one stage and read in a later stage requires a read-write view even if an individual kernel sees only one direction.

## Aliasing boundary

Algorithm alias policy belongs to CUDA-Algorithms. Allocation/view byte-range truth belongs to CUDA-JS.

CUDA-JS `cuda-js@0.1.0-alpha.20` exposes the consumer-neutral public relation:

```text
inspectDeviceViewRelation(a, b)
  -> "same-range" | "overlap" | "disjoint"
```

The relation uses private lower parent/range truth, exposes no allocation/native identity and performs no native/actor work. Invalid/incomparable capabilities fail closed in the lower owner.

The first CUDA-Algorithms Candidate policy is:

```text
read + read:
  same-range / overlap / disjoint are allowed when the family permits reuse

any pair where at least one role writes:
  relation must be disjoint
```

`same-range` or `overlap` with a writing role rejects before prepared algorithm submission.

This policy covers exact same-object aliases, same-range sibling views, containment and partial overlap without CUDA-Algorithms knowing or exposing parent allocation identity.

CUDA-Algorithms must not build a private parent-token registry, deep-import CUDA-JS internals, or expose native addresses to implement alias policy.

## Bounded resources and realization limits

Every plan exposes finite logical workspace/capacity requirements relevant to callers. Provider/runtime implementation ceilings must be identified as realization limits rather than mathematical limits.

Example: current stable multiword ordering uses one reset node plus one prepared kernel node per key word. Under the accepted CUDA-JS 32-node prepared-DAG profile, that realization admits at most 31 words per submitted plan. This does **not** define the semantic maximum width of lexicographic keys.

No hidden queue, scratch growth, retry loop or allocation may be unbounded.

## Determinism

The first candidate integer profile requires exact deterministic outputs and, where specified by the family, stable relative ordering.

Provider-private work assignment, block size, digit width, internal materialization and later accelerators may vary without changing the accepted semantic result.

Floating-point reduction/reassociation policy is outside this Candidate.

## Evidence supporting Candidate status

The common contract has been exercised by two maintained algorithm families and mapped to materially different consumers (BSFP-style record/index processing, CUDA-DATA-style row/column processing, and graph/frontier processing).

Portable evidence against CUDA-JS `98e2ebc942c14d63acf4dd82e912dd548c363a05` / `cuda-js@0.1.0-alpha.20` / Node `v26.7.0` includes:

- 23/23 deterministic reference tests;
- 6/6 maintained Candidate API tests;
- accepted Device-JS inspection for the current status/ordering kernels;
- public CUDA-JS prepared-DAG composition;
- same-range write-conflict rejection;
- partial sibling-view overlap rejection through `inspectDeviceViewRelation`;
- legal read/read reuse;
- physical qualification harness syntax validation.

The historical falsifier, ownership routing and #260 resolution are recorded in `docs/evidence/2026-09-09-device-view-alias-boundary.md`.

This is **not** native CUDA-Algorithms result evidence.

## Acceptance gate

Before this specification becomes Accepted:

1. run the maintained physical qualification harness on an exact directly accessible CUDA profile using the exact CUDA-JS revision under test;
2. compare produced results against the independent reference semantics;
3. prove operation/plan/runtime cleanup on the same run;
4. review the resulting public surface, alias policy and realization bounds at the exact tested revision.

Performance is separately gated. A correctness pass does not justify throughput claims.

## Change rule before acceptance

Candidate names/layouts may still break when evidence identifies a correctness, ownership, capacity, lifecycle or materially better LEGO boundary. Do not add compatibility shims for alpha-only mistakes without a real external beneficiary.
