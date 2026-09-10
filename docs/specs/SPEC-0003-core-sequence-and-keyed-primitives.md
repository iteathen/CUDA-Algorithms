# SPEC-0003: Core Sequence and Keyed GPU Primitives

**Status:** Working Draft
**Date:** 2026-09-09
**Issue:** #3
**Depends on:** SPEC-0001 and working SPEC-0002

> This is a deliberately revisable first production-family draft. Names, exact type coverage, resource formulas and even the primitive decomposition may change while reference and Device-JS implementations are being built.

## Outcome

Define the first reusable primitive spine needed by materially different CUDA-JS ecosystem consumers without importing tensor, dataframe, graph, search or BSFP semantics.

The current candidate family is:

```text
scan
reduce
select indices by flag
stable order indices by external key
gather by index
run-length encode
reduce by key
```

The initial draft used `radixSortKeys` / `radixSortPairs` as the primary ordering surface. Reference work showed a more consumer-neutral public seam: keep records/columns in owner storage and order an index sequence by an external primitive key view. Direct key/pair sorting remains a valid internal realization or later convenience API, but is no longer the primary first-profile semantic.

These primitives are intended to compose explicitly. The first profile does not hide `order + group + reduce` behind a single semantic operation.

## Common data model

Operations consume contiguous one-dimensional CUDA-JS device views plus SPEC-0002 active extents.

The first profile prefers primitive scalar columns and **index-sequence/permutation operations** over a new record/struct ABI.

Candidate storage types:

```text
counts/indices: u32 | u64
ordering keys:  u32 | u64
flags:          u32, values restricted to 0 or 1 where flag semantics apply
```

Other CUDA-JS view dtypes may be admitted later where semantics and provider behavior are exact.

An index value identifies an element in consumer-owned storage. CUDA-Algorithms owns the generic sequence/permutation operation, not the semantic meaning of the referenced element.

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
- insufficient output capacity yields explicit semantic capacity failure, never a success-shaped truncated payload;
- when the exact required output count can be computed safely, capacity failure should retain that required count for administration while payload validity is false.

This index-first design lets arbitrary consumer-owned records/columns remain in place and be materialized later only when needed.

## 4. Stable order-indices by external key

The primary first-profile ordering candidate operates on an existing index sequence rather than owning or moving the consumer's records.

Candidate operation:

```text
stableOrderIndicesByKey({
  keys,
  indicesIn,
  indicesOut,
  active,
  keyCapacity,
  direction?,
  beginBit?,
  endBit?
})
```

For each active sequence position `i`:

```text
record = indicesIn[i]
key(i) = selectedBits(keys[record])
```

The result is `indicesIn[0..active)` stably ordered by `key(i)`.

First candidate key types:

```text
u32 | u64
```

First candidate index types:

```text
u32 | u64
```

Requirements:

- every consumed index must be within `keyCapacity` before the key is dereferenced;
- invalid device-resident indices produce explicit semantic failure and cannot cause unchecked out-of-range device access;
- ascending is the current default candidate;
- optional bit range is half-open `[beginBit, endBit)` and validated against key width;
- ordering is unsigned numeric/radix ordering of the selected key bits;
- ordering is **stable with respect to the current input sequence**: equivalent keys preserve their relative positions from `indicesIn`; numeric index value is not the tie breaker;
- `indicesOut` is a permutation of the active `indicesIn` values when the input itself contains a valid sequence; duplicate index values are not silently normalized away;
- no input/output overlap is assumed until a separately proven ping-pong/in-place profile defines it;
- provider-private digit width, gather strategy, scratch representation and pass structure are not public semantics.

### Why this replaced pair sorting as the primary public seam

The semantic question upper consumers repeatedly ask is not necessarily “move these key/value pairs.” It is often:

> order this logical item sequence by a key stored with the items.

That shape survives deletion of the motivating consumers:

- BSFP can order candidate/proof-record IDs without moving wide records;
- CUDA-DATA can order row IDs by a column without CUDA-Algorithms owning table/row schema;
- graph algorithms can order item/vertex IDs by an external property without graph meaning moving downward.

An implementation remains free to realize this operation as:

```text
indirect compare/order
```

or:

```text
gather primitive keys
  -> provider radix sort of key/index pairs
  -> retain ordered indices
```

or another qualified strategy. The public semantic therefore does not force the provider to perform random indirect accesses if materializing keys is faster.

### Wide fixed keys

Stable index ordering lets consumers construct exact lexicographic ordering of wider fixed keys without CUDA-Algorithms owning a record format.

For a consumer-owned key represented most-significant-first as columns `(w0, w1, ... wN)`, start with an index sequence and apply stable ordering from the least-significant word to the most-significant word:

```text
indices
  -> stableOrderIndicesByKey(wN)
  -> ...
  -> stableOrderIndicesByKey(w0)
```

The final index sequence is lexicographically ordered by the full consumer key.

Reference qualification has already shown this permutation-first formulation equivalent to both the earlier gather+pair-sort composition and an independently implemented tuple-lexicographic oracle on deterministic duplicate-heavy fixtures. Native performance remains unproved.

## 5. Gather by index

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

Gather becomes an explicit materialization boundary. Consumers need not gather merely to carry an ordering when the ordered index sequence itself is sufficient.

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

- runs are maximal adjacent ranges of exactly equal primitive keys;
- one unique key and one run length are emitted per run;
- no sorting is implied;
- global deduplication is obtained only when a caller first establishes the required ordering/equality grouping;
- output run count is device-chainable;
- run lengths use an explicitly selected width sufficient for the active-capacity bound;
- insufficient output capacity cannot turn a partial prefix into a valid result.

The first draft keeps direct primitive-key RLE simple. An indexed-key RLE may later be justified if repeated key materialization proves materially wasteful across consumers.

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

- reduction occurs over maximal adjacent runs of equal primitive keys;
- no sorting is implied;
- key equality is exact for the declared primitive key type;
- first accepted reduction operators should be associative integer operations with exact empty/run initialization behavior;
- output order follows run order.

As with RLE, an indirect/indexed form is a later optimization/generalization question, not assumed merely because ordering is permutation-first.

## Direct key/pair sorting

`radixSortKeys` and `radixSortPairs` remain valid algorithm territory but are no longer required to be the first public ordering studs.

They may appear as:

- private realization mechanisms;
- provider-adapter operations;
- later convenience APIs with independently useful consumer evidence.

Do not expose a CUB-shaped public API merely because CUB is a likely accelerator.

## Hashes are not equality

A consumer may use a `u32`/`u64` hash as an ordering/grouping key, but CUDA-Algorithms must not treat equal hashes as exact record equality.

Exact canonicalization of wider consumer records requires full equality within hash-equivalent candidate groups before identities are collapsed. A hash remains a partition/grouping hint, not identity authority.

## Explicit composition examples

### Stable compaction of consumer records

```text
flags
  -> selectIndices
  -> keep the index sequence
  -> gather only columns/records that must be materialized
```

### Row/item ordering without record movement

```text
existing indices
  -> stableOrderIndicesByKey(external key column)
  -> ordered indices
```

### Wide-key exact ordering

```text
indices
  -> stable order by least-significant word
  -> ...
  -> stable order by most-significant word
```

### Primitive-key unique / aggregation

Where a consumer needs direct grouped keys:

```text
ordered indices
  -> gather key
  -> runLengthEncode / reduceByKey
```

A later indexed grouping family is justified only if this materialization boundary is measured as a real repeated cost.

## Resource and workspace contract

Each primitive must provide or resolve a finite workspace requirement before execution from material plan facts such as dtype, capacity, active-extent form and selected semantic options.

Implementations may use ping-pong buffers, gathered key scratch, prefix intermediates or temporary histograms internally, but requirements must remain bounded and visible through plan/resource metadata rather than hidden unbounded allocation.

Provider-private workspace formulas may differ if both satisfy the same public semantics and are represented honestly in execution planning.

## Determinism

The first integer primitive profile targets exact deterministic semantic outputs.

Stable order/select behavior is part of the result, not merely a performance preference.

Provider tuning may change internal work assignment or materialization strategy but may not change accepted exact integer outputs or stable relative ordering.

## Lower CUDA-JS capability assessment

Correctness-first versions may be implemented with already accepted Device-JS, views and multiple kernel boundaries.

The first scan/select GPU-facing experiment has already passed current CUDA-JS Device-JS frontend inspection and public prepared-DAG mock composition without shared memory, local arrays or warp primitives. That proves expressivity/orchestration only, not native result correctness or performance.

High-performance implementations are still expected to assess the minimum demand-driven subset of CUDA-JS SPEC-0022. Likely candidates remain:

```text
fixed-size local arrays
typed static/dynamic shared memory
selected warp identity/vote/shuffle operations
```

The existing block barrier is already accepted in Device-JS. Do not request a new synchronization abstraction merely because optimized GPU algorithms normally use one.

The first implementation must derive the exact missing capability from code/evidence before a lower CUDA-JS contract is widened.

## Reference qualification

Before Candidate promotion, each selected primitive needs deterministic JavaScript/TypeScript reference cases covering at least:

- empty active extent and one-item inputs;
- duplicate/equal keys;
- all-selected/none-selected/sparse flags;
- boundary key/index values;
- `u32`/`u64` width boundaries selected by the profile;
- stable ordering relative to a non-identity incoming index sequence;
- invalid indirect indices before dereference;
- active extent smaller than capacity;
- device-count overflow/capacity semantic failure models;
- aliasing negatives;
- wide-key composition proving repeated stable index-order passes equal an independent lexicographic oracle.

Native/GPU qualification must compare exact outputs against the independent reference and verify operation/resource terminal cleanup through public CUDA-JS contracts.

## Deferred families

The first accepted slice should not automatically include:

```text
segmented scan/reduce/order
histogram/bucketing
partition families
scatter with conflicting destinations
merge/merge-sort
Top-K
floating reductions
custom comparators/custom reduction callables
generic record/struct ABI
indexed RLE/reduce-by-key
provider-specific CUB/CCCL public surfaces
```

These remain valid future CUDA-Algorithms territory when concrete consumers justify them.

## Mutable-draft rule

Before Candidate promotion, implementation evidence may:

- reduce the first accepted operator/type set;
- split gather or reduce-by-key into later children;
- replace pair-oriented ordering with permutation/index-oriented ordering, as this revision does;
- change exact public naming;
- change workspace planning shape;
- add a missing primitive only if cross-consumer/deletion evidence shows it belongs in the same coherent LEGO.

Do not preserve a bad draft API for compatibility with prototypes.

## Falsifiers

Rework or split this profile if:

- permutation-first ordering causes unavoidable pathological access/materialization costs across materially different consumers and a pair/direct sequence abstraction is demonstrably better;
- stable ordering cannot efficiently support intended wide-key composition;
- device-resident active extents force host synchronization between primitives;
- safe indirect key/gather bounds cannot be enforced through selected lower contracts;
- a primitive requires domain-specific record/schema/proof meaning;
- one primitive's lifecycle/resource model is materially different enough to warrant a separate LEGO/specification family.
