# SPEC-0003: Stable Index Selection and Permutation Ordering

**Status:** Candidate
**Date:** 2026-09-09
**Issue:** #3
**Depends on:** SPEC-0001 and Candidate SPEC-0002

> This revision intentionally contracts the original broad primitive draft to the two families that survived reference work, Device-JS composition and maintained implementation. Scan/reduce/gather/RLE/reduce-by-key remain CUDA-Algorithms territory but are not promoted by this specification.

## Outcome

Define the first maintained reusable CUDA-Algorithms primitive family:

```text
stable select indices by u32 flags
stable lexicographic order of an index sequence by external u32 key-word columns
```

The public shape is deliberately permutation/index oriented. Consumer-owned records, rows, graph objects and proof objects stay where their natural owner stores them.

## Common model

The Candidate operates on public contiguous one-dimensional CUDA-JS `u32` device views and one device-resident `u32` active count under SPEC-0002.

An index value identifies an element in consumer-owned storage. CUDA-Algorithms owns only the generic sequence/permutation transformation.

No Tensor shape, table schema, graph meaning, proof-record meaning, hash identity or consumer equality semantics are inferred.

## 1. Stable select-indices by flag

Maintained plan constructor:

```text
createStableSelectIndicesU32Plan(runtime, {
  inputCapacity,
  outputCapacity,
  blockSize?
})
```

Submission binds:

```text
flags:         u32[inputCapacity] read
prefix:        u32[inputCapacity] read-write workspace
activeCount:   u32[1]             read
outputIndices: u32[outputCapacity] write
outputCount:   u32[1]             write/readable by later consumers
status:        u32[1]             read-write
```

Flag semantics:

- active flags are exactly `0` or `1`;
- result contains source indices whose flag is `1`;
- result order is stable ascending source position;
- zero active items is valid;
- inactive capacity tail has no semantic effect;
- invalid flag/extent becomes explicit semantic failure;
- insufficient output capacity reports capacity failure and, where safely known, retains the required count; partial output is not a valid success result.

The correctness-first realization uses a reset → exclusive flag-prefix computation → stable emission prepared DAG. Its current work complexity is not a performance claim.

## 2. Stable lexicographic order-indices by external key words

Maintained plan constructor:

```text
createStableLexicographicOrderIndicesU32Plan(runtime, {
  recordCapacity,
  indexCapacity,
  keyWordCount,
  blockSize?
})
```

Submission binds:

```text
keyWords:   keyWordCount read-only u32[recordCapacity] views,
            ordered most-significant word to least-significant word
indicesA:   u32[indexCapacity] read-write
indicesB:   u32[indexCapacity] read-write
activeCount:u32[1] read
status:     u32[1] read-write
```

Semantics:

For active sequence position `i`, `indicesA[i]` identifies a record. Its complete ordering key is the tuple:

```text
(keyWords[0][record], keyWords[1][record], ... keyWords[N-1][record])
```

The result is the active incoming index sequence stably ordered lexicographically by that unsigned tuple.

Requirements:

- every active index is checked against `recordCapacity` before key dereference;
- equivalent complete keys preserve relative order from the incoming index sequence;
- duplicate index values remain duplicate values; ordering does not deduplicate;
- inactive tail has no semantic effect;
- invalid extent/index becomes explicit semantic failure;
- `resultBinding` identifies whether `indicesA` or `indicesB` contains the final active permutation after the prepared pass sequence.

The current realization performs one stable pass per key word from least-significant word to most-significant word. Provider implementations may use a different exact strategy.

## Why permutation-first is normative

The original Working Draft centered direct key/key-value sorting. Reference experiments showed the more stable LEGO seam is:

> order this item-index sequence by keys stored with the items.

That allows:

- proof/state engines to order IDs without moving wide records;
- data systems to order row IDs without moving whole rows;
- graph algorithms to order vertex/item IDs without graph meaning entering this library.

A provider may internally gather primitive keys and use a pair sorter, but provider shape does not define public semantics.

## Wide-key composition

Stable per-word passes are exact lexicographic composition, not a hash approximation.

Reference evidence compared permutation-first ordering against both:

- the earlier gather + stable pair-sort composition; and
- an independently implemented tuple-lexicographic oracle.

Duplicate-heavy deterministic fixtures matched exactly.

Hashes may be used by consumers as hints, but equal hashes are never exact record identity.

## Equality and grouping boundary

This Candidate does not own arbitrary record equality or canonicalization.

If an ordered index sequence is later grouped, the natural consumer owns exact adjacent equality/change predicates. CUDA-Algorithms may consume generic flags/indices/scan mechanics without redefining:

- BSFP proof-record equality;
- CUDA-DATA null/collation semantics;
- graph/domain equality.

The grouping-boundary reference fixtures showed the same generic segmentation mechanics working for both wide record equality and data-style null equality while leaving equality with the consumer.

## Aliasing

CUDA-Algorithms owns whether its logical roles may overlap; CUDA-JS owns the underlying byte-range relation.

The Candidate consumes public CUDA-JS:

```text
inspectDeviceViewRelation(a, b)
  -> "same-range" | "overlap" | "disjoint"
```

Policy:

- read-only roles may share or overlap ranges when the family permits read reuse;
- if either role writes, the two ranges must be `disjoint`;
- `same-range` or `overlap` with a writing role rejects before prepared submission;
- lower relation errors for stale/cross-runtime/incomparable views propagate fail-closed.

For ordering this means, among other cases:

- `indicesA` and `indicesB` must be disjoint ranges;
- key ranges must be disjoint from mutable index/status ranges;
- repeated or overlapping read-only key ranges are legal.

For selection, mutable prefix/output/control ranges must be disjoint from every conflicting role; read/read reuse remains legal.

Portable Candidate evidence exercises both exact same-range conflicts and partially overlapping sibling views.

## Bounds

All capacities and physical launch dimensions are finite and validated before lower execution.

The current ordering realization uses:

```text
1 reset node + keyWordCount ordering nodes
```

Accepted CUDA-JS prepared execution currently admits 32 nodes, therefore this realization exposes:

```text
maxKeyWordCount = 31
```

This is an implementation/profile bound, not a semantic claim that keys wider than 31 words are unsupported in principle.

## Device-JS realization

The correctness-first kernels use accepted CUDA-JS mechanisms only:

- restricted Device-JS;
- public typed device views;
- `gpu.thread.globalX`;
- `gpu.atomic.cas`;
- `gpu.atomic.loadRelaxedDevice` for coherent same-location status observation;
- prepared kernel DAG execution.

No shared-memory, warp, local-array, CUDA C++, PTX-maintained source or private lower import is required for correctness.

Broader Device-JS parallel helpers remain potential **performance** work and must be justified by measurements rather than assumed necessary.

## Determinism

For valid integer inputs the semantic outputs are exact and deterministic.

Stable relative ordering is part of the contract.

CUDA thread scheduling, physical stream overlap, block size and provider-private strategy may not change the result.

## Candidate evidence

Portable qualification against CUDA-JS `98e2ebc942c14d63acf4dd82e912dd548c363a05`, `cuda-js@0.1.0-alpha.20`, Node `v26.7.0`:

- 23/23 independent/reference semantic tests passed;
- 6/6 maintained Candidate API tests passed;
- Device-JS frontend inspection passed;
- public prepared-DAG composition passed;
- stable multiword/duplicate-heavy ordering matched independent oracles;
- same-range write-conflict rejection passed;
- partially overlapping sibling-view rejection passed through lower-owned range truth;
- legal read/read reuse passed;
- physical maintained-Candidate qualification harness is syntax-valid.

The historical alias falsifier and CUDA-JS #260 resolution are recorded in `docs/evidence/2026-09-09-device-view-alias-boundary.md`.

Frontend/mock identity is not native result evidence.

## Native acceptance gate

Before this specification becomes Accepted:

1. run the maintained physical harness using the exact CUDA-JS revision under test on a directly accessible CUDA profile;
2. compare selection prefix/count/indices/status and multiword ordering results against independent references;
3. include invalid extent/flag/index and capacity-pressure cases;
4. require graceful operation/plan/runtime cleanup;
5. review the final public surface and alias policy at the tested revision.

## Deferred algorithm families

Not promoted by this Candidate:

```text
general scan/reduce public APIs
gather/scatter
partition
run-length encode
reduce-by-key
histogram/bucketing
segmented operations
Top-K
floating reductions
generic record/struct ABI
custom comparator/reducer callables
provider-specific CUB/CCCL public APIs
```

They remain candidate CUDA-Algorithms ownership only when consumer evidence justifies activation.

## Change rule before acceptance

Break Candidate names/layouts if native evidence, ownership, resource truth, aliasing, or performance qualification exposes a materially better complete design. Do not preserve alpha-only mistakes through compatibility shims without a real beneficiary.
