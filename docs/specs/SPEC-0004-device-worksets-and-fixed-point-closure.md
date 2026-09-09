# SPEC-0004: Device Worksets and Fixed-Point Closure

**Status:** Working Draft
**Date:** 2026-09-09
**Issue:** #3
**Depends on:** SPEC-0001 and working SPEC-0002; may consume working SPEC-0003 primitives

> This specification is intentionally more exploratory than SPEC-0003. It defines the reusable semantic target for GPU-owned progression while leaving callback/composition details open until the first vertical slice is implemented.

## Outcome

Define a provider-neutral bounded workset/closure abstraction for algorithms that repeatedly derive, group, merge, filter and advance GPU-resident work without requiring Node to perform mathematical progression.

The abstraction must be useful beyond any one game/search/data/graph consumer.

## Ownership

CUDA-Algorithms owns generic:

- workset/frontier meaning;
- bounded capacity and active-count semantics;
- ranked acyclic progression;
- monotone workset/fixed-point progression;
- convergence and budget-yield semantics;
- algorithm-level epoch/admin boundary meaning;
- deterministic plan/resource contracts for the above.

Consumers retain:

- record/domain meaning;
- transition or derivation semantics;
- proof/search/table/graph semantics;
- terminal/domain predicates;
- domain-specific ordering or conflict policy.

CUDA-JS retains execution, compilation, synchronization, memory and native lifecycle mechanisms.

## Workset model

The first candidate workset is deliberately index-oriented:

```text
Workset
  itemIndices: device view<u32|u64>
  activeCount: device-resident extent
  capacity:    host-known finite bound
```

The indices identify records/state owned by the consumer or by another semantic layer. CUDA-Algorithms does not own the structure of those records merely because it schedules their indices.

An implementation may use additional internal key/index/status buffers, but public workset meaning remains independent of physical layout.

## Progression building blocks

A generic workset step conceptually consists of some subset of:

```text
DERIVE / EXPAND
  -> VALIDATE / FILTER
  -> KEY
  -> ORDER / GROUP
  -> MERGE / REDUCE
  -> DELTA / DOMINANCE
  -> COMPACT
  -> NEXT WORKSET
```

Not every consumer needs every stage.

CUDA-Algorithms may own reusable implementations of the generic ordering/grouping/compaction stages. Consumer-specific derivation/equality/dominance meaning remains outside unless a lower reusable algebra is independently established.

## Callback/composition boundary is intentionally open

This Working Draft does **not** yet freeze whether consumer-defined work is expressed as:

- statically linked Device-JS library functions;
- caller-supplied CUDA-JS function capabilities in an accepted prepared DAG;
- a bounded declarative transform contract;
- another consumer-neutral composition mechanism.

The first BSFP/CUDA-DATA/graph prototypes must determine which option preserves static typing, resource ownership, compilation identity and GPU autonomy with the least accidental complexity.

No dynamic device function pointers or arbitrary CUDA callback escape are implied.

## Ranked closure

Ranked closure is the first preferred activation target because it has a simpler correctness and resource model than arbitrary cyclic fixed points.

A RankedClosure plan has a finite rank domain and the invariant that every derived dependency moves monotonically in the declared direction.

Conceptually:

```text
rank N
  -> rank N-1
  -> ...
  -> rank 0
```

or the equivalent increasing orientation.

Requirements:

- rank order is explicit and finite;
- no derivation may create a dependency that violates the declared rank monotonicity;
- violation is a semantic error/falsifier, not silently inserted into another rank;
- a completed rank may be compacted/checkpointed according to consumer/admin policy after every dependency requiring it is satisfied;
- local batch/shard counts do not limit total logical problem size;
- the mathematical result is independent of how a rank is partitioned into bounded physical batches, provided the declared merge/canonicalization semantics are satisfied.

BSFP is the strongest first consumer, but no CPC/WSL-625/NDC/WDL vocabulary belongs in RankedClosure.

## General monotone workset closure

A later or parallel profile may support cyclic/unranked monotone closure.

Abstractly, let `S` be accumulated facts/state and `F` a consumer-supplied monotone derivation under an accepted merge/order relation.

The closure target is a fixed point satisfying:

```text
S* = merge(S*, F(S*))
```

The exact least/greatest/order-theoretic interpretation belongs to the specific accepted profile and consumer-supplied algebra. CUDA-Algorithms must not claim generic fixed-point correctness without explicit monotonicity/order/merge preconditions.

The device workset contains only newly relevant work (`delta`) where the selected algorithm permits semi-naive/delta progression.

## GPU ownership rule

For a profile claiming GPU-owned progression:

> After an epoch is submitted, every decision that changes mathematical workset progression is device-owned until a declared administrative boundary.

Node may not inspect individual records or counts to decide what mathematically survives or what dependency is expanded next.

Node may perform bounded administration:

- submit/prepare operations through CUDA-JS;
- asynchronously observe a device-produced status;
- supply a next opaque input shard requested by the device plan;
- persist an opaque completed output/checkpoint region;
- reclaim/rebind resources only when lower lifecycle contracts permit it;
- resubmit the next bounded epoch after a device-produced administrative yield.

No synchronous Node event-loop wait is permitted in the GPU-owned production path.

## Epochs and administrative boundaries

The closure engine may use bounded epochs rather than one unbounded persistent kernel.

Candidate device-produced administrative states include:

```text
running
converged
rank-complete
needs-input
needs-spill
capacity-yield
budget-yield
invalid-semantic-state
failed
```

Exact spelling/layout is not frozen.

An administrative yield is not mathematical convergence. A budget limit, watchdog-safe kernel boundary, spill boundary or unavailable input must never be reported as a fixed point.

## Bounded work

Every plan declares finite bounds for material resources such as:

```text
workset capacity
candidate capacity
per-item maximum emission where applicable
key/index width
workspace bytes
rank range or epoch work budget
number of prepared nodes/operations per epoch where material
status/control storage
```

A capacity or work-budget boundary must produce explicit backpressure/yield/failure truth rather than truncating, dropping work or falsely converging.

## Partition and shard invariance

A major goal is to support finite problems larger than available VRAM.

The semantic result of an accepted ranked/workset algorithm must not depend on arbitrary physical shard size.

CUDA-Algorithms may own the logical requirements needed to prove shard equivalence, such as:

- complete partition coverage;
- deterministic merge/canonicalization across shard outputs;
- stable global key/order requirements where material;
- generation/version identities for restart/checkpoint correctness.

CUDA-MM owns reusable physical spill/eviction/migration/prefetch policy if activated. `cuda-io` owns persistent source/sink mechanics. CUDA-Algorithms must not become either subsystem.

## Exact canonicalization

If a closure profile performs duplicate elimination, equality authority must be exact.

A hash may partition candidate work but cannot be the sole equality criterion for exact closure.

Where consumer records are wider than primitive keys, a valid realization may use:

```text
hash/group hint
  -> exact full equality within candidate groups
  -> canonical ID / merge
```

or stable multiword ordering followed by exact adjacent comparison.

The equality/canonical-record semantics remain with the natural owner unless CUDA-Algorithms later establishes a truly generic fixed-record contract.

## Determinism and scheduling

Logical results must not depend on CUDA thread scheduling, stream overlap or physical kernel concurrency unless the accepted semantic contract explicitly permits nondeterminism.

Physical overlap is a performance property, not a correctness prerequisite.

A provider may choose different batch, tile or kernel strategies while preserving the same accepted workset/closure result and administrative-state semantics.

## Relationship to CUDA-JS prepared execution

Current CUDA-JS prepared execution can represent finite reusable kernel DAGs and return one ordinary CUDA operation for a submitted DAG. That is a promising epoch-realization mechanism, but CUDA-Algorithms must not assume CUDA Graph realization or dynamically repeating graphs exist until their lower contracts are accepted/qualified.

A first implementation may therefore use finite prepared/multi-operation epochs with asynchronous Node administration between epochs, provided the device decides the mathematical state and no host semantic step is required.

## Cooperative/grid-wide execution

CUDA-JS issue #223 separately assesses cooperative launch/grid synchronization.

Closure design must not assume cooperative groups, grid-wide barriers or persistent cooperative residency in its semantics. These may become optional performance realizations only after ordinary multi-kernel correctness exists and measurements demonstrate material benefit.

## Out-of-core checkpoint semantics

The first ranked closure implementation should make completed bounded units independently persistable where practical.

A logical checkpoint/shard descriptor may include algorithm-owned facts such as:

```text
algorithm/plan identity
logical rank/partition identity
input generation/digest references
output record count
canonicalization generation
semantic completion state
```

Physical file format, filesystem policy and transfer mechanism are not owned here unless separately generalized into their natural libraries.

A restart must never treat a partially written/semantically incomplete shard as complete merely because bytes exist.

## Reference and qualification

Before RankedClosure becomes Candidate:

1. implement a small exact reference model over ordinary JavaScript data;
2. prove partition/shard invariance on finite fixtures;
3. prove device-count chaining without host count decisions;
4. integrate at least two materially different consumer models;
5. run one real CUDA-JS GPU vertical slice where the host performs administration only;
6. compare every produced closure/result against an independent consumer/reference oracle;
7. exercise capacity and budget-yield paths without false convergence;
8. verify lower CUDA-JS operation/resource cleanup.

General monotone closure requires additional evidence for monotonicity, convergence and cyclic/worklist behavior and need not block RankedClosure acceptance.

## Mutable-draft rule

While this specification remains Working Draft:

- callback/composition API may change substantially;
- ranked and monotone closure may split into separate specs;
- workset physical representation may change;
- status/admin state vocabulary may change;
- checkpoint descriptor shape may change;
- ranked closure may be accepted first while general closure remains draft;
- any prototype must record the exact draft/commit it tested.

Do not add compatibility shims for prototype-only names or layouts.

## Falsifiers

Rework this abstraction if:

- GPU-owned progression requires CUDA-Algorithms to own native CUDA scheduling mechanisms;
- the callback/composition boundary cannot remain statically bounded and consumer-neutral;
- ranked shard partition changes exact mathematical results under otherwise valid execution;
- device-produced administrative state is insufficient to keep Node out of the semantic loop;
- workset meaning becomes inseparable from graph/search/proof/table semantics;
- closure requires unbounded hidden queues or silently dropped work;
- an ordinary lower-level primitive composition is simpler and equally reusable, making a standalone closure abstraction unnecessary.
