# Device-view alias ownership boundary evidence

**Date:** 2026-09-09
**Status:** Durable design/qualification evidence; not specification authority

## Question

Can CUDA-Algorithms enforce a consumer-level non-alias contract using only the current public CUDA-JS device-view and prepared-DAG surfaces?

## Exact lower authority exercised

- CUDA-JS commit: `e9837f20acf7901d445a1e7a2045459a1ae0118a`
- package identity: `cuda-js@0.1.0-alpha.19`
- Node: `v26.7.0`
- CUDA-Algorithms branch: `design/first-algorithm-profile`
- cleanup-safe falsifier commit: `f5477e5afe0f2f87fc0171a0c986acac0f5e5264`
- GitHub Actions run: `34423412784`

The run used the public CUDA-JS testing facade and prepared-DAG surface. This is portable orchestration evidence, not native GPU evidence.

## Result

The falsifier intentionally bound the same public `u32` device view to the stable-selection experiment as both:

- `flags`: read role; and
- `prefix`: write role.

CUDA-JS accepted the prepared submission. The test therefore failed with the intended diagnostic:

> CUDA-JS prepared submission accepted a same-view read/write alias inside one algorithm node

The surrounding reference, maintained-candidate, Device-JS inspection and ordinary prepared-composition checks remained green.

## Interpretation

This does **not** contradict the accepted CUDA-JS prepared-DAG contract. CUDA-JS owns concrete view/range validation and operation/DAG hazard ordering. CUDA-Algorithms owns whether a particular algorithm permits its logical roles to alias within one kernel/algorithm invocation.

The public `CudaDeviceView` intentionally does not expose its parent allocation/native identity. Therefore an upper library can directly recognize an exact same JavaScript view object, but it cannot reliably determine whether two different public sibling views overlap the same underlying allocation.

That is the actual missing stud.

## LEGO disposition

### CUDA-Algorithms owns

- whether an algorithm permits input/output/workspace/control aliasing;
- fail-closed admission when an observable relation violates that policy;
- keeping read/read reuse legal when the algorithm permits it.

### CUDA-JS owns

- allocation/view range truth;
- parent/runtime/device/generation identity;
- a consumer-neutral relation between two public view capabilities if that relation is exposed.

CUDA-Algorithms must not recover private parent identity, import lower internals, or invent a second allocation registry merely to enforce alias policy.

## Immediate candidate behavior

The maintained candidate API rejects **exact same-view conflicts when either role writes**. Pure read/read reuse remains legal.

This is deliberately narrower than a full non-overlap guarantee. Two distinct sibling views that overlap cannot yet be classified by CUDA-Algorithms from public facts alone.

The broader relation is tracked as:

- `iteathen/CUDA-JS#260` — **Device views: expose consumer-neutral range relation for upper-layer alias validation**.

Until a consumer-neutral lower relation exists and is qualified, CUDA-Algorithms must not claim that it detects every overlapping sibling-view conflict.

## Why no CUDA-JS workaround was copied here

A local wrapper registry, hidden parent token, pointer comparison, private import, or native escape would violate LEGO ownership and SPEC-0001. The correct response is to protect what can be proven locally and route the missing generic fact to its natural lower owner.

## Qualification consequence

This evidence supports Candidate-level development of the first primitive surface, but full alias-contract acceptance remains scoped:

- exact same-view write conflicts: enforceable now;
- pure read/read same-view reuse: permitted;
- distinct overlapping sibling views: pending CUDA-JS #260 or an equally consumer-neutral public mechanism;
- native CUDA-Algorithms result correctness: separately unqualified until the maintained physical harness is run.
