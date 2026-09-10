# Core Reference Qualification — First Primitive Draft

**Date:** 2026-09-09
**Branch:** `design/first-algorithm-profile`
**Reference implementation commit:** `ceafc950a7f639fa908f08298d6da7a482251ef7`
**Environment:** Node v22.16.0 local reference-only execution
**Result:** 15/15 tests passed

## Scope

This qualification exercises the deterministic JavaScript reference semantics for the initial SPEC-0002/SPEC-0003 Working Draft subset.

It does **not** establish CUDA-JS package compatibility, native GPU support, provider support, or performance. CUDA-JS source development currently requires a newer Node profile; Node v22.16.0 is recorded only as the environment in which this standalone reference oracle executed.

## Exercised semantics

- inclusive and exclusive `u32` scan;
- explicit `u32` modular-add behavior;
- `u64` arithmetic beyond JavaScript safe-integer Number precision;
- active extent smaller than storage capacity;
- stable flag-to-index selection;
- capacity exhaustion with no truncated valid payload;
- flag validation;
- gather ordering, duplicate source indices and out-of-range rejection;
- stable radix ordering for duplicate keys;
- selected radix bit-range semantics;
- adjacent-only run-length encoding;
- run-length capacity exhaustion;
- adjacent-run reduce-by-key;
- repeated stable word-pass composition versus an independently implemented lexicographic oracle;
- 100 deterministic duplicate-heavy wide-key fixtures, 1–6 words and 1–64 rows each;
- device-active-extent validation model.

## Material findings

### 1. Stable primitive-word sorting remains a strong generalization seam

Repeated stable least-significant-word passes produced the same index ordering as an independent tuple-lexicographic oracle on every exercised fixture.

This supports keeping the first radix primitive limited to primitive keys plus associated indices instead of inventing a generic consumer-record ABI.

### 2. Reference unsigned semantics should use BigInt

The reference implementation uses `BigInt` for both `u32` and `u64` value semantics. This avoids accidental JavaScript 32-bit bitwise coercion and preserves `u64` values beyond Number safe-integer precision.

This is reference/oracle policy only. It does not prescribe the GPU physical representation.

### 3. Capacity exhaustion should not produce a success-shaped truncated result

For selection and run-length encoding fixtures, an insufficient output capacity returns an explicit `capacity-exhausted` state and no valid partial payload. The exact required output count remains available where it can be computed safely.

This is a useful candidate refinement for the Working Draft status contract because it lets administration resize/retry without turning a truncated prefix into semantic data.

### 4. Hashes remain grouping hints, never equality authority

The reference work did not need hashes to prove wide-key ordering. This reinforces the current rule that a future hash-assisted canonicalization path must still perform exact equality before collapsing records.

## Remaining uncertainty

- exact public API naming and plan construction;
- physical device status/control record layout;
- device-resident active-count chaining through real CUDA-JS execution;
- workspace formulas;
- which primitive/type subset should be the first Candidate profile;
- exact lower Device-JS helper gap for efficient scan/radix/select;
- callback/composition boundary for RankedClosure;
- native GPU correctness and performance.

## Next falsifier

Construct the smallest exact GPU vertical slice through public CUDA-JS using current accepted mechanisms first. Prefer an intentionally simple correctness implementation over premature shared-memory/warp optimization. The purpose of that slice is to discover the real lower capability and API seams, then revise the Working Drafts before Candidate promotion.
