# CUDA-Algorithms Status

**Updated:** 2026-09-09

**Architecture/ownership:** accepted independent reusable GPU parallel-algorithm semantic owner under ADR-0001 and SPEC-0001.
**Production implementation/API:** not yet authorized; issue #3 activation design is in progress.
**Provider/support:** none selected or claimed.

## Current work

- #1 ownership/bootstrap authority — completed through PR #4.
- #2 repository-control/protected-main alignment — partially complete. `main` is protected; remaining repository-setting/code-owner parity debt is administrative and does **not** block issue #3 design/reference work.
- #3 first consumer-backed algorithm profile — **active** on `design/first-algorithm-profile`.

## Issue #3 activation state

The first working design intentionally separates three LEGOs:

1. common algorithm-plan, active-extent and device-chaining semantics;
2. reusable sequence/keyed primitives;
3. GPU-owned workset/fixed-point closure.

Current Working Draft specifications:

- SPEC-0002 — Algorithm Plans, Active Extents, and Device Chaining;
- SPEC-0003 — Core Sequence and Keyed GPU Primitives;
- SPEC-0004 — Device Worksets and Fixed-Point Closure.

These drafts are **not compatibility authority**. During the first implementation cycle, exact names, type coverage, status layout, workspace shape and even spec decomposition may change when evidence shows a better design. Prototype evidence must record the exact draft revision it tested.

## Selected first primitive spine

Current candidate set:

```text
scan
reduce
stable select-indices by flag
gather by index
stable radix sort keys / key-index pairs
run-length encode
reduce-by-key
```

The design uses index indirection before inventing a generic record ABI. Stable radix ordering is normative so wider consumer-owned keys can be composed from repeated stable primitive-word passes.

## Device-owned progression target

A device-resident active count plus explicit capacity should let one GPU stage feed another without Node reading counts between stages.

For GPU-owned closure profiles, Node may administer allocations, submission, persistence/checkpoints and asynchronous status observation, but it must not perform mathematical frontier selection, grouping, deduplication, predecessor decisions or fixed-point advancement on the CPU.

Ranked acyclic closure is the preferred first closure slice. General cyclic/monotone workset closure may remain draft longer if its callback/composition boundary is not yet stable.

## CUDA-JS capability assessment

Accepted CUDA-JS already supplies restricted Device-JS, typed contiguous views, atomics, block barrier/device fence, bounded multi-operation execution, async transfers, prepared kernel DAG semantics and typed Device-JS library composition.

The broader trusted Device-JS parallel proposal still leaves fixed local arrays, typed shared memory and warp primitives demand-driven/proposal-only. Correctness-first CUDA-Algorithms implementations should use current accepted capabilities first. Only the minimum generic lower mechanism demonstrated necessary by implementation/performance evidence should be routed to CUDA-JS.

CUDA-JS #223 separately owns cooperative-launch/grid-sync assessment and is not a prerequisite for the first correctness path.

## Next executable work

1. Add deterministic JavaScript reference semantics for the first primitive subset.
2. Add property/edge fixtures for stability, width, capacity, active extent and multiword stable-sort composition.
3. Build the smallest exact GPU vertical slice through public CUDA-JS without CPU semantic work.
4. Record the first concrete lower CUDA-JS capability gap rather than assuming the whole SPEC-0022 parallel family is required.
5. Revise Working Drafts immediately when the implementation exposes a better abstraction.
6. Promote only the stable subset to Candidate, then Accepted after exact qualification.

## Protected-main readback

`main` is protected by active default-branch integrity and PR-review rulesets. Remaining #2 parity debt is tracked separately and must not be confused with algorithm implementation readiness.

No required status-check name is fabricated because CUDA-Algorithms has no local CI workflow yet.

## Ecosystem boundaries

CUDA-JS owns generic Device-JS/compiler/runtime/memory/provider/lifecycle mechanisms. CUDA-JS-Tensor owns Tensor mathematics. CUDA-DATA owns tables/columns/dataframes. CUDA-GRAPH-ANALYTICS owns graph-analysis meaning. CUDA-MM owns generic physical placement/spill policy if activated. CUDA-MCGS and downstream solvers retain search/proof/domain semantics.

No Working Draft, prototype, provider availability or first-consumer success is production support/performance authority.
