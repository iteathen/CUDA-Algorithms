# Exact Grouping and Canonicalization Ownership Boundary

**Status:** active Working Draft design evidence
**Date:** 2026-09-09
**Issue:** #3

## Question

After permutation-first ordering, should CUDA-Algorithms own a generic exact-record canonicalizer/equality callback, or should exact equality remain with the consumer while CUDA-Algorithms supplies only reusable sequence/segmentation mechanics?

Current answer: **keep exact equality with the natural consumer owner.**

## Why

Equality is often not a generic GPU-algorithm fact.

- BSFP owns exact proof/dependency-record identity. A hash is not equality.
- CUDA-DATA owns table/key semantics such as null treatment, collation and categorical meaning.
- CUDA-GRAPH-ANALYTICS owns graph-object/property meaning where equality depends on graph semantics.

Moving these facts into CUDA-Algorithms would violate LEGO ownership even if the implementation happened to be reusable.

## Reusable lower algebra

After a consumer has produced a correctly ordered index sequence, it can produce a bounded binary `changeAfter` sequence:

```text
changeAfter[i] = 1  iff item i and item i+1 are not exactly equal
changeAfter[last] = 0
```

The equality predicate used to create these flags remains consumer-owned.

Generic group IDs then follow from ordinary sequence algebra:

```text
groupId = exclusiveScan(changeAfter, add, init=0)
```

Example:

```text
ordered items:   A A B B B C
changeAfter:     0 1 0 0 1 0
exclusive scan:  0 0 1 1 1 2
```

The same flags can locate segment boundaries/representatives with select/gather composition or a future independently justified segment primitive.

## Why this is better than a generic canonicalizer callback

A generic CUDA-Algorithms `canonicalize(records, equalityCallback)` would force this layer to own or dynamically compose consumer equality semantics, callback ABI, record layout and potentially consumer lifecycle/resource facts.

The flag boundary avoids that contamination:

```text
consumer exact equality kernel
          |
          v
     change flags
          |
          v
CUDA-Algorithms scan/select/gather
          |
          v
 generic segment/group structure
```

CUDA-Algorithms still provides the reusable expensive sequence machinery while the semantic owner keeps the fact only it can define correctly.

## Reference falsifier

A local deterministic reference experiment tested:

1. wide multiword record keys representative of BSFP-style exact structural identity; and
2. data-style equality where null rows compare equal despite irrelevant underlying payload values.

For each case, consumer-owned exact equality generated `changeAfter`; an exclusive additive scan generated group IDs; an independent grouping oracle generated group IDs directly from neighboring exact values.

Result:

```text
501 / 501 fixtures matched
```

The wide-record set covered 500 deterministic duplicate-heavy fixtures with 1–10 key words and 1–100 items each. One separate data-style fixture exercised caller-owned null equality/canonical-key semantics.

This is reference/design evidence only; no native GPU performance or API claim follows.

## Consequence for SPEC-0003

Do **not** add a general consumer-record equality/canonicalization API to the first primitive profile.

Keep:

- stable index ordering;
- scan;
- select/gather;
- direct primitive-key RLE/reduce-by-key where primitive equality is fully owned by CUDA-Algorithms.

For wider/domain-sensitive exact canonicalization, consumers produce exact boundary/change flags. A future generic segment-head/offset convenience operation is allowed only if it has independent reusable semantics and does not absorb consumer equality.

## Important distinction

This decision does not forbid CUDA-Algorithms from owning primitive equality where the contract itself fully defines it, such as exact `u32`/`u64` run-length encoding.

It forbids promoting **consumer record identity** into this repository merely because grouping machinery runs here.

## Falsifier

Revisit this boundary only if materially different consumers demonstrate that consumer-produced boundary flags create substantial unavoidable duplication or prevent an efficient GPU-only path, and a consumer-neutral statically bounded equality/key contract can be specified without importing record/domain semantics.
