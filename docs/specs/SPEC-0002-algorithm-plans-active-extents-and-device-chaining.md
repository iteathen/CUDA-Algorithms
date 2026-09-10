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
- algorithm-specific workspace and realization bounds.

CUDA-JS owns:

- allocations, device views and their private parent/range truth;
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

Algorithm alias policy belongs to CUDA-Algorithms; underlying allocation/view relation truth belongs to CUDA-JS.

The maintained candidate can currently prove and enforce:

- exact same public view object + at least one writing role => reject before algorithm submission;
- exact same public view object used only by read roles => allowed when the family permits it.

Current public CUDA-JS views intentionally hide parent allocation identity. Therefore two distinct sibling views cannot yet be classified by CUDA-Algorithms as disjoint/overlapping/contained.

The missing consumer-neutral relation is tracked by `iteathen/CUDA-JS#260`. Until that or an equivalent lower capability is accepted and consumed, the first CUDA-Algorithms candidate must **not** claim full detection of overlapping sibling-view aliases.

CUDA-Algorithms must not build a private parent-token registry, deep-import CUDA-JS internals, or expose native addresses to work around this boundary.

## Bounded resources and realization limits

Every plan exposes finite logical workspace/capacity requirements relevant to callers. Provider/runtime implementation ceilings must be identified as realization limits rather than mathematical limits.

Example: current stable multiword ordering uses one reset node plus one prepared kernel node per key word. Under the accepted CUDA-JS 32-node prepared-DAG profile, that realization admits at most 31 words per submitted plan. This does **not** define the semantic maximum width of lexicographic keys.

No hidden queue, scratch growth, retry loop or allocation may be unbounded.

## Determinism

The first candidate integer profile requires exact deterministic outputs and, where specified by the family, stable relative ordering.

Provider-private work assignment, block size, digit width, internal materialization and later accelerators may vary without changing the accepted semantic result.

Floating-point reduction/reassociation policy is outside this Candidate.

## Evidence supporting Candidate status

The common contract has now been exercised by two maintained algorithm families and mapped to materially different consumers (BSFP-style record/index processing, CUDA-DATA-style row/column processing, and graph/frontier processing).

Portable evidence against CUDA-JS `e9837f20acf7901d445a1e7a2045459a1ae0118a` / Node `v26.7.0` includes:

- 23/23 deterministic reference tests;
- 5/5 maintained candidate API tests;
- accepted Device-JS inspection for the current status/ordering kernels;
- public CUDA-JS prepared-DAG composition;
- explicit exact-view write-conflict and legal read/read alias tests;
- physical qualification harness syntax validation.

The alias falsifier and ownership disposition are recorded in `docs/evidence/2026-09-09-device-view-alias-boundary.md`.

This is **not** native CUDA-Algorithms result evidence.

## Acceptance gate

Before this specification becomes Accepted:

1. run the maintained physical qualification harness on an exact directly accessible CUDA profile;
2. compare produced results against the independent reference semantics;
3. prove operation/plan/runtime cleanup on the same run;
4. resolve or explicitly narrow the distinct-sibling-view alias contract using CUDA-JS #260 or equivalent evidence;
5. review the resulting public surface and realization bounds at an exact revision.

Performance is separately gated. A correctness pass does not justify throughput claims.

## Change rule before acceptance

Candidate names/layouts may still break when evidence identifies a correctness, ownership, capacity, lifecycle or materially better LEGO boundary. Do not add compatibility shims for alpha-only mistakes without a real external beneficiary.
