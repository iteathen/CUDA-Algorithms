# First Production Profile Design — GPU-Resident Parallel Algorithms

**Status:** Working design; intentionally mutable during first implementation experiments
**Date:** 2026-09-09
**Issue:** #3

## Purpose

Select the first reusable CUDA-Algorithms shape from real consumer pressure without freezing an API before implementation evidence exists.

This document is design evidence, not accepted production authority. The first implementation experiments are expected to change names, record layouts, helper composition, limits, and possibly the decomposition itself. Changes are encouraged while the design is still cheap to correct. Accepted specifications are the compatibility boundary; working drafts are not.

## Governing architecture

CUDA-Algorithms is the algorithm-semantic LEGO above CUDA-JS mechanisms and below consumer/domain semantics.

```text
BSFP / CUDA-DATA / CUDA-GRAPH-ANALYTICS / other consumers
                         |
                         v
                  CUDA-Algorithms
   sequence + ordering + keyed + workset/closure algorithms
                         |
                         v
                      CUDA-JS
 Device-JS + views + compilation + execution + transfers + lifecycle
```

Optional future composition:

```text
CUDA-Algorithms -> CUDA-MM     physical placement/spill policy
CUDA-Algorithms -> cuda-io     persistence/source/sink mechanics
```

Neither optional layer owns algorithm meaning.

## Design hierarchy applied here

1. Preserve exact algorithm semantics and domain-independent ownership.
2. Keep device progression possible without a host data-decision loop.
3. Use domain-appropriate widths, capacities, lifecycle and failure truth.
4. Expose small public studs that survive deletion of BSFP or any other first consumer.
5. Prefer explicit composition over convenience operations that hide major work or duplicate ownership.
6. Optimize only after exact reference behavior exists.

## Consumer pressure

### BSFP / NDC

Needs a GPU-resident pipeline approximately shaped as:

```text
produce candidate records
  -> compute structural keys
  -> order/group
  -> exact-equivalence handling
  -> deduplicate/reduce
  -> compact surviving work
  -> advance ranked closure
```

BSFP additionally requires that Node perform administration only. Node must not inspect proof records, choose predecessor semantics, deduplicate records, or advance the mathematical fixed point on the CPU.

### CUDA-DATA

Needs the same lower algebra for filtering, stable compaction, ordering and keyed aggregation without importing dataframe/schema semantics into CUDA-Algorithms.

### CUDA-GRAPH-ANALYTICS

Needs bounded frontier/workset and fixed-point progression without importing BFS/SSSP/component/PageRank meaning into CUDA-Algorithms.

These three consumers are materially different enough to be a useful LEGO/deletion test.

## Selected decomposition

### 1. Common plan and chaining substrate

Algorithms consume ordinary public CUDA-JS device-memory/view capabilities plus explicit logical roles and active extents. CUDA-Algorithms does not invent a second memory system.

An active extent can eventually be either:

- host-known, validated before submission; or
- device-resident, bounded by a host-known capacity so one GPU result can feed the next GPU operation without a D2H count round trip.

This is important enough to be a separate cross-cutting contract rather than rediscovered independently by scan, sort and closure.

### 2. Sequence and keyed primitives

The first candidate primitive spine is deliberately smaller than CUB/Thrust or a general STL:

```text
inclusive/exclusive scan
reduce
stable select-indices / compaction support
gather by index
stable radix sort keys/pairs
run-length encode
reduce-by-key
```

Deferred until consumer evidence justifies them:

```text
segmented families
scatter with duplicate-destination policy
histogram
merge
Top-K
adjacent difference
full comparator-based merge sort
arbitrary transforms/for-each
```

The current NVIDIA CUB device-wide family is useful external evidence that scan/reduce/select/radix-sort/run-length/keyed primitives form a reusable GPU layer, but CUB is not semantic authority for this project.

## Index indirection before generic records

The first profile should avoid inventing a struct/record ABI that CUDA-JS does not own.

Instead, values carried through ordering/selection are initially indices (`u32` or `u64`) into consumer-owned storage. Arbitrary fixed-width or structure-of-arrays records can move with `gather` after their indices have been ordered or selected.

This has several advantages:

- consumer records remain consumer-owned;
- CUDA-Algorithms does not need tensor or struct semantics;
- sorting one compact index payload is cheaper than trucking wide records through every radix pass;
- multi-column CUDA-DATA and wide BSFP records use the same mechanism;
- exact record equality can remain independent of a hash key.

## Stable ordering as a foundational requirement

The first radix-sort profile should be stable.

Stability is not cosmetic. It lets a consumer construct exact lexicographic ordering of arbitrary fixed-width multiword keys by repeated least-significant-word stable passes while the library itself only understands primitive `u32`/`u64` key words and associated indices.

A hash may be used to partition or cheaply group candidates, but a hash match must never be treated as exact equality. Exact consumers such as BSFP must resolve collisions with full structural equality before canonical IDs are assigned.

## Dynamic active extent and no host advancement loop

The first design should not require every algorithm to know `numItems` on the host.

A device-resident active count with an explicit maximum capacity allows:

```text
GPU producer
  -> device count
  -> scan/select/sort/etc.
  -> device count
  -> next GPU stage
```

without Node reading counts between stages.

If a device count exceeds its declared capacity, execution must fail semantically on-device or yield an explicit overflow/backpressure state; it must never silently truncate.

Host code may asynchronously observe completion/progress for administration, persistence or recovery, but host observation does not define or drive the algorithm result.

## Closure/workset layer

The distinctive higher reusable layer is a bounded GPU workset/closure engine built from the primitive spine.

Two modes are intentionally separate:

### Ranked closure

For acyclic problems with a monotonic rank/dependency order:

```text
rank N -> rank N-1 -> ... -> root
```

This is the natural BSFP path and also fits dynamic-programming/dataflow families.

### Monotone workset closure

For cyclic or unordered monotone problems:

```text
frontier -> derive candidates -> merge/delta -> new frontier -> repeat
```

until no new information exists or a declared execution/resource budget yields an administrative boundary.

The general mode must not be forced into the first implementation if ranked closure proves the cleaner first consumer-backed slice.

## Node / GPU ownership rule

For a GPU-owned profile, after submission every decision that changes mathematical progression is device-owned until a declared administrative boundary.

Node may:

- allocate through public CUDA-JS APIs;
- compile/prepare/submit operations;
- asynchronously observe status;
- persist or restore opaque checkpoint/shard bytes;
- respond to a device-produced administrative state such as `complete`, `needs-input`, `needs-spill`, `budget-yield`, or `failed`.

Node must not:

- inspect individual algorithm records to decide progression;
- compute keys, scans, reductions or duplicate groups;
- choose which mathematical frontier survives;
- perform a CPU fixed-point advancement step;
- synchronously block the event loop waiting for GPU progress.

## Current CUDA-JS fit

Accepted CUDA-JS already provides the important lower foundations:

- restricted Device-JS with typed arithmetic, global/thread identity, atomic add/CAS, block barrier and device fence;
- bounded typed contiguous device views;
- bounded multi-operation scheduling and asynchronous transfers;
- semantic prepared-kernel DAG execution;
- typed Device-JS library composition;
- accepted device-scope atomic observation/publication children.

The broader trusted Device-JS parallel profile still leaves fixed local arrays, typed shared-memory views and warp primitives as proposal-only/demand-driven work. CUDA-Algorithms is now a concrete consumer that may justify a minimum child slice, but the exact lower requirement should be derived from the first implementation rather than pre-authorizing the whole proposal.

CUDA-JS issue #223 already owns assessment of cooperative launch/grid synchronization. Do not make cooperative execution a prerequisite until measurements show kernel boundaries are a material bottleneck.

## First realization strategy

### Phase A — exact reference semantics

Implement small JavaScript reference functions for the selected primitive semantics. These are qualification oracles, not a CPU production backend.

### Phase B — correctness-first Device-JS realizations

Use only already accepted public CUDA-JS capabilities to construct the simplest exact GPU path, even if it is multi-pass/global-memory heavy.

This phase answers what the actual API and intermediate state need to be.

### Phase C — lower-capability extraction

If performance or boundedness exposes a real generic missing mechanism, route only that mechanism to CUDA-JS. Likely candidates include static shared memory, fixed local arrays and selected warp helpers, but no one of these is assumed required until evidence demonstrates it.

### Phase D — provider/performance profiles

Only after semantics and API shape stabilize should optional CCCL/CUB-backed or other provider realizations be assessed. Provider availability must not redefine the public contract.

## Specification-change policy during activation

Working drafts may change freely on the feature branch while implementation/reference evidence is being gathered.

A useful maturity sequence is:

```text
Working Draft
  -> Candidate
  -> Accepted
```

- **Working Draft:** exploratory; no compatibility obligation; rename/split/delete freely when evidence improves the design.
- **Candidate:** intended production shape; reference semantics and at least two materially different consumer mappings exist; changes remain allowed but must reconcile dependent evidence.
- **Accepted:** production semantic authority; compatibility and evolution rules apply.

This policy is intentionally asymmetric: it is cheap to change the specification early and expensive to preserve a bad early abstraction forever.

## Primary falsifiers

Reconsider this decomposition if any of the following occurs:

- the same public primitive requires BSFP-specific concepts to be useful;
- CUDA-DATA or graph analytics require a materially different lower abstraction rather than the selected primitive semantics;
- device-resident active extents cannot compose safely through public CUDA-JS contracts;
- stable index-based ordering cannot support exact wide-record canonicalization without pathological extra movement;
- closure semantics require owning consumer state/lifecycle that properly belongs elsewhere;
- maintaining the abstraction forces CUDA-Algorithms to invent CUDA/native/provider semantics;
- CPU semantic participation is required for correctness rather than administration;
- a simpler decomposition preserves all required semantics and substantially lowers lifecycle/resource/implementation complexity.

## Immediate implementation seam

1. Draft common plan/active-extent semantics.
2. Draft the first primitive semantics.
3. Draft ranked/workset closure semantics separately so primitive implementation does not depend on premature closure details.
4. Implement reference semantics and property fixtures.
5. Implement the smallest exact GPU vertical slice through public CUDA-JS.
6. Revise working specs based on the first divergence between design and reality.
7. Promote only the stable subset to Candidate, then Accepted after qualification.
