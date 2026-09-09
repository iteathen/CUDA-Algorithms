# SPEC-0003: Core Sequence and Keyed GPU Primitives

**Status:** Working Draft
**Date:** 2026-09-09
**Issue:** #3
**Depends on:** SPEC-0001 and working SPEC-0002

> This is a deliberately revisable first production-family draft. Names, exact type coverage and resource formulas may change while reference and Device-JS implementations are being built.

## Outcome

Define the first reusable primitive spine needed by materially different CUDA-JS ecosystem consumers without importing tensor, dataframe, graph, search or BSFP semantics.

The first candidate family is:

```text
scan
reduce
select indices by flag
gather by index
stable radix sort keys / key-index pairs
run-length encode
reduce by key
```

These primitives are intended to compose explicitly. The first profile does not hide `sort + group + reduce` behind a single semantic operation.

## Common data model

Operations consume contiguous one-dimensional CUDA-JS device views plus SPEC-0002 active extents.

The first profile prefers primitive scalar columns and index indirection over a new record/struct ABI.

Candidate storage types:

```text
counts/indices: u32 | u64
radix keys:     u32 | u64
flags:          u32, values restricted to 0 or 1 where flag semantics apply
```

Other CUDA-JS view dtypes may be admitted later where semantics and provider behavior are exact.

## 1. Scan

Candidate operation:

```text
scan({
  input,
  output,
  active,
  mode: "inclusive" | "exclusive",
  op,
  init?
})
```

First candidate operators for unsigned integer inputs:

```text
add
min
max
bitAnd
bitOr
bitXor
```

Requirements:

- operator is associative over the declared dtype semantics;
- output order corresponds exactly to input order;
- zero active items is valid;
- exclusive scan has one explicit initial value;
- unsigned addition follows the declared integer dtype arithmetic; consumers using scan for counts must choose width/capacity so valid counts cannot wrap;
- in-place support is not assumed until separately proven and stated;
- device-resident active extent must be supported by any profile claiming device chaining.

The first implementation may narrow to `add` if that materially reduces lower-capability work; other listed operators remain draft candidates until implemented and qualified.

## 2. Reduce

Candidate operation:

```text
reduce({ input, outputScalar, active, op, init })
```

The first integer operator family mirrors scan where useful.

Requirements:

- exact identity/empty-input behavior is explicit through `init`;
- operator requirements are stated per dtype;
- floating-point reduction is deferred from the first accepted profile unless a grouping/determinism policy is explicitly specified and qualified;
- no provider may change result semantics by choosing a different unsupported reduction order for a non-associative operator.

## 3. Stable select-indices by flag

Instead of defining generic record compaction first, the initial primitive emits indices for selected input positions.

Candidate operation:

```text
selectIndices({ flags, outputIndices, outputCount, active, indexType })
```

Semantics:

- one flag corresponds to each active input position;
- accepted flag values are exactly `0` or `1` in the first profile;
- output contains the indices whose flags are `1`;
- output order is stable ascending source order;
- output count is device-chainable under SPEC-0002;
- insufficient output capacity yields explicit semantic capacity failure, never truncation.

This index-first design lets arbitrary consumer-owned records/columns be compacted later by gather without CUDA-Algorithms owning their schema.

## 4. Gather by index

Candidate operation:

```text
gather({ input, indices, output, activeIndices })
```

Semantics:

```text
output[i] = input[indices[i]]
```

Requirements:

- every consumed index must be in range for the bound input logical capacity;
- invalid device-resident indices must produce explicit semantic failure rather than unchecked out-of-range memory access;
- output preserves index order exactly;
- duplicate source indices are valid;
- input/output aliasing is rejected in the first profile unless a later in-place contract proves a safe case;
- supported payload dtype is a property of the exact accepted profile, not inferred from consumer schema.

The first implementation should prioritize the CUDA-JS scalar view dtypes required by the initial consumers rather than promise the full dtype registry prematurely.

## 5. Stable radix sort

Candidate operations:

```text
radixSortKeys({ keysIn, keysOut, active, direction?, beginBit?, endBit? })

radixSortPairs({
  keysIn,
  keysOut,
  indicesIn,
  indicesOut,
  active,
  direction?,
  beginBit?,
  endBit?
})
```

First candidate key types:

```text
u32 | u64
```

First candidate associated value types:

```text
u32 | u64 index values
```

Requirements:

- ascending default unless the accepted profile chooses otherwise;
- optional bit range is half-open `[beginBit, endBit)` and validated against key width;
- ordering is radix/unsigned numeric ordering of the selected key bits;
- sort is **stable**: items with equivalent selected key bits preserve prior relative order;
- key-index association is preserved exactly;
- no input/output overlap unless a separately accepted double-buffer/in-place profile defines it;
- provider-private digit width/pass strategy is not public semantics.

### Why stability is normative

Stable primitive-word sorting lets consumers construct exact lexicographic ordering of wider fixed keys by repeated least-significant-word passes.

For example, a consumer-owned key represented as words `(w0, w1, ... wN)` can be ordered without CUDA-Algorithms owning that record format by stable sorting associated indices from least-significant word to most-significant word.

This is a general composition mechanism for wide structural keys, not a BSFP-specific feature.

## 6. Run-length encode

Candidate operation:

```text
runLengthEncode({
  keys,
  uniqueKeys,
  runLengths,
  outputRunCount,
  active
})
```

Semantics:

- runs are maximal adjacent ranges of exactly equal keys;
- one unique key and one run length are emitted per run;
- no sorting is implied;
- global deduplication is obtained only when a caller first establishes the required ordering/equality grouping;
- output run count is device-chainable;
- run lengths use an explicitly selected width sufficient for the active-capacity bound.

## 7. Reduce by key

Candidate operation:

```text
reduceByKey({
  keys,
  values,
  uniqueKeys,
  aggregates,
  outputRunCount,
  active,
  op,
  initPolicy
})
```

Semantics:

- reduction occurs over maximal adjacent runs of equal keys;
- no sorting is implied;
- key equality is exact for the declared primitive key type;
- first accepted reduction operators should be associative integer operations with exact empty/run initialization behavior;
- output order follows run order.

## Hashes are not equality

A consumer may use a `u32`/`u64` hash as a radix key to cheaply group candidate records, but CUDA-Algorithms must not treat equal hashes as exact record equality.

Exact canonicalization of wider consumer records requires a full equality check within hash-equivalent groups before identities are collapsed. That equality may later be expressed through a consumer-supplied bounded callable or a separate fixed-record primitive, but hash collision handling is not optional for exact consumers.

## Explicit composition examples

### Stable compaction of consumer records

```text
flags
  -> selectIndices
  -> gather(record column A)
  -> gather(record column B)
  -> ...
```

### Primitive-key unique

```text
radixSortPairs
  -> runLengthEncode
```

### Keyed aggregation

```text
radixSortPairs
  -> reduceByKey
```

### Wide-key exact ordering

```text
stable radix pass on least-significant word
  -> ...
  -> stable radix pass on most-significant word
```

## Resource and workspace contract

Each primitive must provide or resolve a finite workspace requirement before execution from material plan facts such as dtype, capacity, active-extent form and selected semantic options.

Implementations may use ping-pong/double buffers, prefix intermediates or temporary histograms internally, but these requirements must remain bounded and visible through plan/resource metadata rather than hidden unbounded allocation.

Provider-private workspace formulas may differ if both satisfy the same public bounds/semantics and are represented honestly in execution planning.

## Determinism

The first integer primitive profile targets exact deterministic semantic outputs.

Stable sort/select order is part of the result, not merely a performance preference.

Provider tuning may change internal work assignment but may not change accepted exact integer outputs or stable relative ordering.

## Lower CUDA-JS capability assessment

Correctness-first versions may be implemented with already accepted Device-JS, views and multiple kernel boundaries.

High-performance implementations are expected to assess the minimum demand-driven subset of the CUDA-JS SPEC-0022 trusted parallel proposal. Likely candidates are:

```text
fixed-size local arrays
typed static/dynamic shared memory
selected warp identity/vote/shuffle operations
```

The existing block barrier is already accepted in Device-JS. Do not request a new synchronization abstraction merely because shared-memory algorithms normally use one.

The first implementation must derive the exact missing capability from code/evidence before a lower CUDA-JS contract is widened.

## Reference qualification

Before Candidate promotion, each selected primitive needs deterministic JavaScript/TypeScript reference cases covering at least:

- empty, one-item and maximum-small fixture sizes;
- duplicate/equal keys;
- all-selected/none-selected/sparse flags;
- boundary key values;
- `u32`/`u64` width boundaries selected by the profile;
- stable ordering with repeated equal keys;
- active extent smaller than capacity;
- device-count overflow/capacity semantic failure models;
- aliasing negatives;
- wide-key composition proving repeated stable passes equal an independent lexicographic oracle.

Native/GPU qualification must compare exact outputs against the independent reference and verify operation/resource terminal cleanup through public CUDA-JS contracts.

## Deferred families

The first accepted slice should not automatically include:

```text
segmented scan/reduce/sort
histogram/bucketing
partition families
scatter with conflicting destinations
merge/merge-sort
Top-K
floating reductions
custom comparators/custom reduction callables
record/struct ABI
provider-specific CUB/CCCL public surfaces
```

These remain valid future CUDA-Algorithms territory when concrete consumers justify them.

## Mutable-draft rule

Before Candidate promotion, implementation evidence may:

- reduce the first accepted operator/type set;
- split gather or reduce-by-key into later children;
- change exact public naming;
- change workspace planning shape;
- add a missing primitive only if two-consumer/deletion evidence shows it belongs in the same coherent LEGO.

Do not preserve a bad draft API for compatibility with prototypes.

## Falsifiers

Rework or split this profile if:

- index indirection produces unacceptable required data movement for materially different consumers;
- stable radix semantics cannot efficiently support the intended wide-key composition;
- device-resident active extents force host synchronization between primitives;
- safe gather bounds cannot be enforced through the selected lower contracts;
- a primitive requires domain-specific record/schema/proof meaning;
- one primitive's lifecycle/resource model is materially different enough to warrant a separate LEGO/specification family.
