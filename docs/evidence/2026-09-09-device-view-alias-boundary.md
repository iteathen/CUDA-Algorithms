# Device-view alias ownership boundary evidence

**Date:** 2026-09-09
**Status:** Durable design/qualification evidence; lower relation resolved on CUDA-JS main; not specification authority

## Question

Can CUDA-Algorithms enforce a consumer-level non-alias contract using only public CUDA-JS device-view capabilities without importing private allocation identity?

## Phase 1 — falsifier before CUDA-JS #260

Exact lower authority:

- CUDA-JS commit: `e9837f20acf7901d445a1e7a2045459a1ae0118a`
- package identity: `cuda-js@0.1.0-alpha.19`
- Node: `v26.7.0`
- CUDA-Algorithms cleanup-safe falsifier commit: `f5477e5afe0f2f87fc0171a0c986acac0f5e5264`
- GitHub Actions run: `34423412784`

The falsifier bound one public `u32` view simultaneously as:

- `flags`: read role; and
- `prefix`: write role.

CUDA-JS prepared submission accepted the binding. The test failed with the intended diagnostic:

> CUDA-JS prepared submission accepted a same-view read/write alias inside one algorithm node

This did **not** contradict the prepared-DAG contract. CUDA-JS owned concrete view/range validity and DAG hazard ordering; CUDA-Algorithms owned whether its logical roles were permitted to alias.

The real missing stud was that public views hid parent allocation identity, so an upper library could recognize the exact same JavaScript object but could not classify two different sibling views as same-range, overlap or disjoint.

## LEGO disposition

### CUDA-Algorithms owns

- whether an algorithm permits input/output/workspace/control ranges to overlap;
- fail-closed algorithm admission when a prohibited relation is observed;
- allowing read/read reuse where the algorithm permits it.

### CUDA-JS owns

- allocation/view byte-range truth;
- parent/runtime/device/generation identity;
- the consumer-neutral relation between two public view capabilities.

CUDA-Algorithms must not recover private parent identity, import CUDA-JS internals, maintain a shadow allocation registry, or expose native addresses to enforce its policy.

## Phase 2 — CUDA-JS #260 resolved

CUDA-JS issue #260 was implemented and merged by PR #261 at:

```text
98e2ebc942c14d63acf4dd82e912dd548c363a05
cuda-js@0.1.0-alpha.20
```

The public stud is:

```text
inspectDeviceViewRelation(a, b)
  -> "same-range" | "overlap" | "disjoint"
```

The lower implementation validates live same-runtime public capabilities using private parent/range truth, performs no actor/native call for the relation, exposes no parent/native identity, and fails closed for incomparable/stale/closing/closed capabilities.

CUDA-JS keeps the **fact** neutral. It does not decide whether an upper algorithm admits the relation.

## Current CUDA-Algorithms Candidate policy

The maintained Candidate now consumes `inspectDeviceViewRelation` directly.

For every pair of logical roles:

```text
read + read:
  overlap/same-range may be admitted by the algorithm

anything involving a write:
  relation must be disjoint
  same-range or overlap => reject before prepared submission
```

This covers both exact same-object aliases and distinct sibling views that partially overlap or represent the same byte range.

Portable candidate tests include:

- same-range write-conflict rejection;
- partially overlapping sibling-view rejection;
- legal read/read reuse;
- ordinary disjoint bindings.

## Why the ownership split matters

The solution did not widen CUDA-Algorithms with lower memory identity and did not widen CUDA-JS with algorithm policy.

```text
CUDA-JS:        what is the byte-range relation?
CUDA-Algorithms: is that relation legal for this algorithm?
```

That is the intended LEGO boundary.

## Qualification consequence

The previous alias blocker for Candidate acceptance is now resolved at the public-contract level.

Remaining acceptance work is primarily:

- portable requalification against exact CUDA-JS alpha.20;
- maintained CUDA-Algorithms physical result qualification on an exact CUDA profile;
- graceful operation/plan/runtime cleanup evidence;
- final review of the public alias policy at the tested revision.

CUDA-JS's earlier gate-32 Windows physical evidence is tied to its recorded alpha.19 compatible pair. The new alpha.20 view relation is a pure host-side facade relation with no native work; CUDA-Algorithms must still record the exact alpha.20 revision used by its own physical run rather than silently transferring an old package qualification.
