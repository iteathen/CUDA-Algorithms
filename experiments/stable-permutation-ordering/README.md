# Stable Permutation Ordering Slice

**Status:** experiment / Working Draft evidence only
**Issue:** #3

## Question

Can CUDA-Algorithms treat ordering as **stable permutation of an index sequence by an external key view**, keep consumer records in place, and compose multiword lexicographic order entirely on device-facing CUDA-JS contracts?

## Correctness-first kernel

`stableOrderIndicesByKeyU32` is deliberately O(N^2). Each active sequence position computes its unique stable rank by counting input positions whose referenced key is smaller, plus earlier positions with an equal key.

Conceptually:

```text
record = indicesIn[i]
key = keys[record]
rank = count(j where key(j) < key || (key(j) == key && j < i))
indicesOut[rank] = record
```

This is not a production ordering algorithm. It exists to test the semantic/API boundary without requiring shared memory, warp primitives or a vendor radix-sort provider first.

## Safety/status

Prototype status values:

```text
0  OK
1  INVALID_EXTENT
2  OUTPUT_CAPACITY_EXHAUSTED
4  INVALID_INDEX
```

Every indirect index is checked against the external key capacity **before** dereference. On semantic failure, any partially written output is invalid and must not be consumed as a result.

## Two-word wide-key composition

The orchestration experiment prepares one DAG:

```text
reset status
  -> stable order by low word:  indicesA -> indicesB
  -> stable order by high word: indicesB -> indicesA
```

The final `indicesA` is the lexicographic order for a two-word most-significant/high + least-significant/low key because each pass is stable.

The same scheme extends to more fixed words by ping-ponging index buffers from least-significant word to most-significant word. The public semantic need not own a record/struct ABI.

## Backend freedom

A future optimized implementation does not have to use indirect comparison. It may gather primitive key values internally and use a qualified radix-sort provider while preserving the same permutation-first public result.

That distinction is intentional: public semantics describe the reusable algorithm; provider mechanics remain replaceable.

## Qualification boundary

The reference implementation has already matched the earlier gather+stable-pair-sort composition and an independent tuple oracle across explicit fixtures plus 250 deterministic duplicate-heavy fixtures.

The next qualification for this directory is CUDA-JS frontend inspection and prepared-DAG portable orchestration. Even if that passes, physical GPU numerical correctness and performance remain separate unproved claims.
