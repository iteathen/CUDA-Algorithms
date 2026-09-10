# Permutation-First Ordering Refinement

**Status:** active Working Draft design evidence
**Date:** 2026-09-09
**Issue:** #3

## Discovery

The initial first-profile draft treated stable `radixSortKeys` / `radixSortPairs` as the primary public ordering operations.

While building the first reference/GPU-facing slices, the stronger reusable question emerged:

> Can consumer-owned records stay in place while CUDA-Algorithms manipulates only a bounded sequence of their indices?

That is a better LEGO boundary for the first profile.

## Revised semantic seam

The current preferred public ordering operation is conceptually:

```text
stableOrderIndicesByKey(keys, indicesIn) -> indicesOut
```

where `keys` is an external primitive key column and `indicesIn` identifies consumer-owned items.

The operation owns generic ordering/permutation meaning only. It does not own what an index identifies or what the keyed object means.

## Why this generalizes better

### BSFP

Candidate/proof records can remain in wide consumer-owned storage while an index sequence is ordered by a selected structural key/hash/word.

### CUDA-DATA

Rows can remain columnar while row IDs are ordered by one column. Materialization of other columns occurs only if/when needed.

### CUDA-GRAPH-ANALYTICS

Item or vertex IDs can be ordered by an external property without moving graph meaning into CUDA-Algorithms.

The same public stud survives deletion of every one of these consumers.

## Backend freedom

Permutation-first **public semantics do not require indirect-memory implementation**.

A provider may implement the operation by:

```text
indirect key reads + ordering
```

or:

```text
gather active primitive keys
  -> radix sort key/index pairs
  -> discard/materialize keys as useful
```

or another exact realization.

This preserves compatibility with future CCCL/CUB acceleration without making a vendor-shaped pair-sorting API the semantic owner.

## Wide-key consequence

Stable ordering is still essential.

For fixed multiword keys held as separate consumer columns, repeatedly stable-order the same index sequence from least-significant word to most-significant word. The final permutation is exact lexicographic order without defining a generic record ABI.

## Reference evidence

A direct permutation-first reference implementation was compared against:

1. the earlier gather + stable pair-sort composition; and
2. an independently implemented tuple-lexicographic oracle.

It matched both on the explicit fixture and on 250 deterministic duplicate-heavy fixtures spanning 1–8 key words and 1–96 records.

This is semantic/reference evidence only, not GPU-performance evidence.

## Spec consequence

SPEC-0003 is revised while still `Working Draft`:

- stable order-indices-by-key becomes the primary first-profile ordering semantic;
- direct key/pair sorting remains possible realization/later convenience territory;
- RLE/reduce-by-key stay direct/simple for now rather than immediately propagating indexed variants;
- gather remains the explicit materialization boundary;
- no compatibility shim is added for the superseded draft surface.

## Remaining performance question

Permutation-first semantics could still be a bad choice if all efficient realizations incur unavoidable pathological materialization/indirection across materially different consumers.

That is now an explicit falsifier. Provider implementation is free to gather keys internally, so a performance failure must implicate the **semantic abstraction**, not merely one indirect kernel implementation.
