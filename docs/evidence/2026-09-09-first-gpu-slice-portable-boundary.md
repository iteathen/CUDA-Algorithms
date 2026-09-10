# First GPU Slice — Portable CUDA-JS Boundary Qualification

**Date:** 2026-09-09
**CUDA-Algorithms branch:** `design/first-algorithm-profile`
**Qualified experiment head:** `3269e1215c6772509bd4e5834915827fcd38230a`
**Pinned CUDA-JS authority:** `97c0295ab79add204d4d8ced080a4da4b66149cf`
**Node:** v26.7.0
**GitHub Actions run:** `34419481929`
**Job:** `102691587884`
**Conclusion:** success

## Scope

This record qualifies the first correctness-first CUDA-Algorithms GPU-facing slice against current public CUDA-JS frontend and portable orchestration contracts.

It is **not native GPU execution evidence**. The GitHub-hosted Ubuntu runner had no qualified CUDA device path; CUDA-JS's `openCudaRuntimeForTesting()` is explicitly a portable lifecycle/orchestration mock.

## Experiment

The experimental stable-selection chain is:

```text
resetControl
  -> exclusiveFlagScanU32
  -> selectIndicesFromScanU32
```

The scan is deliberately O(N^2) and exists to test semantic/device-chaining boundaries before performance work.

The pipeline uses a host-known finite physical capacity and a device-resident active-count input. No host active-count readback is required between the scan and selection stages.

## Exact checks

The successful CI run performed, in order:

1. exact CUDA-Algorithms checkout;
2. exact CUDA-JS checkout at `97c0295ab79add204d4d8ced080a4da4b66149cf`;
3. Node v26.7.0 setup;
4. CUDA-JS dependency installation;
5. the 15-case deterministic JavaScript reference suite;
6. CUDA-JS `inspectDeviceProgram()` on the experimental Device-JS program;
7. a prepared-DAG composition/submission/lifecycle test through the public CUDA-JS testing facade.

All steps passed.

## Device-JS frontend result

CUDA-JS accepted the three kernels under contract:

```text
SPEC-0013-v1+SPEC-0022-atomic-observation-v1+SPEC-0022-device-publication-v1+SPEC-0014-publication-mailbox-v1
```

Normalized Device-JS program identity:

```text
e6671d97e90f8a6c3c2e7d400d3e65f8f6af06316faa63128f67131c50edd1aa
```

Public helper usage was exactly:

```text
exclusiveFlagScanU32:
  gpu.atomic.cas
  gpu.thread.globalX

resetControl:
  gpu.thread.globalX

selectIndicesFromScanU32:
  gpu.thread.globalX
```

No shared-memory, local-array, warp, cooperative-grid, raw CUDA or private-provider capability was required for this correctness-first source.

## Public prepared-DAG result

The portable orchestration test constructed the slice entirely through public CUDA-JS contracts:

```text
compile Device-JS
  -> load public artifact
  -> resolve typed kernel functions
  -> prepare 3-node / 2-edge operation DAG
  -> bind public typed device views
  -> submit one opaque prepared-batch operation
  -> wait in the test harness
  -> terminal cleanup
```

The prepared DAG reported:

```text
nodeCount: 3
edgeCount: 2
realization: semantic-single-stream
bindings:
  activeCount
  flags
  outputCount
  outputIndices
  prefix
  status
```

The mock operation terminalized as a completed `prepared-batch`, and runtime close reported graceful cleanup without restart requirement.

The test harness uses `wait()` because its purpose is qualification. This does not authorize synchronous waiting in a GPU-owned production Node control path; the Working Draft continues to require nonblocking administration there.

## Design consequences

### Supported

- A device-resident active count is compatible with the current Device-JS frontend.
- The first scan/select chain can be represented as one explicit prepared CUDA-JS operation DAG.
- Node does not need to semantically advance between the three stages.
- We do **not** need to widen CUDA-JS merely to obtain a correctness-first implementation of this slice.
- Shared memory, local arrays and warp helpers remain potential optimization mechanisms, not current correctness prerequisites.

### Still unproved

- actual kernel numerical/output correctness on a physical GPU;
- GPU visibility/result behavior beyond what the frontend/mock executes;
- performance;
- optimal scan/select implementation;
- optimized lower-helper requirements;
- zero-physical-capacity representation;
- stable radix sort and keyed primitive device implementations;
- RankedClosure execution;
- out-of-core/shard progression.

## Working-Draft pressure discovered

The experiment currently requires positive physical input/output capacities because the first prepared access plan uses ordinary nonempty buffer ranges. This must **not** silently become a public semantic rule merely because the prototype chose the simplest physical shape.

Before Candidate promotion, decide whether zero logical capacity is represented by:

- a separately normalized zero-work plan;
- minimum physical padding with explicit logical capacity zero; or
- another consumer-neutral CUDA-JS-compatible representation.

Active extent zero within positive capacity is already supported by the semantic design.

## Next falsifier

Move to a second algorithmic shape that stresses ordering/grouping rather than prefix/selection while retaining the same device-resident extent and public CUDA-JS boundaries. Stable integer ordering is the preferred next target because it directly tests the index-indirection/wide-key design that BSFP and CUDA-DATA both need.
