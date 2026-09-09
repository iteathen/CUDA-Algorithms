# First GPU Slice — Device-Count-Driven Stable Selection

**Status:** experiment / Working Draft evidence only
**Issue:** #3

## Question

Can the first CUDA-Algorithms sequence pipeline consume a **device-resident active count** and advance from scan to stable selection without Node reading the count or performing a semantic step between GPU stages?

This experiment deliberately prioritizes correctness and boundary discovery over performance.

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

## What this can prove

If accepted by the real CUDA-JS Device-JS frontend and then executed successfully on a qualified GPU path, the experiment can support these narrow claims:

- current public Device-JS is expressive enough for a correctness-first device-count-driven scan/select chain;
- device-resident active extents do not inherently require a host semantic advancement loop;
- the draft common-plan/device-chaining concept can proceed without first widening CUDA-JS with shared memory or warp operations.

It cannot prove:

- useful scan/select performance;
- production CUDA-Algorithms API shape;
- that shared memory, local arrays, warp operations or another lower helper will not be needed for the optimized implementation;
- arbitrary-size out-of-core closure;
- BSFP correctness or performance.

## Current qualification state

Local ordinary Node syntax checking of the experiment wrapper/source passed.

The local sandbox does not contain the CUDA-JS package and cannot reach GitHub to install/clone it, so `inspectDeviceProgram()` has not yet been executed here. This is an environment limitation, not a Device-JS acceptance result.

Next qualification must use current CUDA-JS under its supported Node source-development profile and run:

```text
node experiments/first-gpu-slice/inspect.mjs
```

The resulting Device-JS contract/identity/helper-usage record should be retained as evidence. Any frontend rejection should be treated as design feedback and the Working Draft/prototype should be corrected rather than compatibility-shimmed.

## Next native boundary

After frontend inspection succeeds, execute the three-stage pipeline through public CUDA-JS operations and compare device `status`, `outputCount`, prefix values and selected indices against `reference/core-primitives.mjs` across empty, sparse, dense, invalid-flag, invalid-extent and insufficient-output-capacity fixtures.

No CPU reference function belongs in the production GPU execution path; it is an independent qualification oracle only.
