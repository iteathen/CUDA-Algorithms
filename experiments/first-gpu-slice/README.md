# First GPU Slice — Device-Count-Driven Stable Selection

**Status:** experiment / Working Draft evidence only
**Issue:** #3

## Question

Can the first CUDA-Algorithms sequence pipeline consume a **device-resident active count** and advance from scan to stable selection without Node reading the count or performing a semantic step between GPU stages?

Current portable answer: **yes at the CUDA-JS frontend/orchestration boundary**. Native GPU execution remains unproved.

## Shape

The Device-JS program contains three kernels:

```text
resetControl
  -> exclusiveFlagScanU32
  -> selectIndicesFromScanU32
```

`exclusiveFlagScanU32` is intentionally O(N^2): each active output thread sums all preceding flags. That is not a proposed production scan implementation. It avoids prematurely requiring shared-memory or warp primitives and lets the current accepted CUDA-JS surface falsify the API/device-chaining design first.

The host launches against a declared maximum capacity. Each semantic kernel reads `activeCount[0]` on device and ignores work items at or above the active extent.

## Device semantic status

The current experimental status values are:

```text
0  OK
1  INVALID_EXTENT
2  OUTPUT_CAPACITY_EXHAUSTED
3  INVALID_FLAG
```

These numeric values and physical layout are prototype details, not accepted public API.

Important behavior:

- `activeCount > inputCapacity` becomes `INVALID_EXTENT`;
- flags must be exactly 0 or 1;
- selected count is written on device;
- insufficient output capacity reports the required selected count but writes no selected-index payload;
- selected indices are stable in ascending source order;
- no D2H active-count read is required between scan and select.

## Portable qualification

Qualified experiment head:

```text
3269e1215c6772509bd4e5834915827fcd38230a
```

Pinned CUDA-JS:

```text
97c0295ab79add204d4d8ced080a4da4b66149cf
```

GitHub Actions run `34419481929`, job `102691587884`, Node v26.7.0: **success**.

The run passed:

- 15/15 deterministic reference tests;
- CUDA-JS `inspectDeviceProgram()` for the three kernels;
- public CUDA-JS testing-facade construction and submission of one prepared 3-node/2-edge DAG;
- graceful mock lifecycle cleanup.

Frontend Device-JS identity:

```text
e6671d97e90f8a6c3c2e7d400d3e65f8f6af06316faa63128f67131c50edd1aa
```

Only `gpu.thread.globalX` and `gpu.atomic.cas` were required by the inspected source. No shared-memory, local-array, warp, cooperative-grid or raw native capability is a correctness prerequisite for this experimental slice.

See `docs/evidence/2026-09-09-first-gpu-slice-portable-boundary.md` for the exact claim boundary.

## Public orchestration experiment

`execution-plan.mjs` constructs the chain only through public `cuda-js` APIs:

```text
compileDeviceProgram
  -> runtime.loadModule
  -> module.getFunction
  -> runtime.prepareOperationDag
```

The prepared DAG exposes named device-memory/view bindings for:

```text
flags
prefix
activeCount
outputIndices
outputCount
status
```

The testing harness waits only to qualify terminal behavior. A production GPU-owned control path must remain nonblocking at the Node event-loop level.

## Known Working-Draft pressure

The current physical plan requires positive input/output capacities because its first access declarations use nonempty ordinary ranges. This is a prototype choice, **not yet a public semantic rule**. Zero logical/physical capacity representation must be resolved before Candidate promotion rather than accidentally inherited from this implementation.

## What remains unproved

- physical GPU result correctness;
- useful scan/select performance;
- production CUDA-Algorithms API shape;
- optimized shared-memory/warp requirements;
- stable radix/keyed device implementations;
- RankedClosure;
- arbitrary-size out-of-core progression.

## Next

The next falsifier is stable integer ordering with device-resident active extent. It should stress the index-indirection/wide-key design using the current accepted CUDA-JS surface before any lower capability widening is requested.
