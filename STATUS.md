# CUDA-Algorithms Status

**Updated:** 2026-09-09

**Architecture/ownership:** accepted independent reusable GPU parallel-algorithm semantic owner under ADR-0001 and SPEC-0001.
**Production implementation/API:** not authorized / none.
**Provider/support:** none selected or claimed.

## Current work

- #1 ownership/bootstrap authority — completed through PR #4; the documentation/ownership foundation is integrated on `main` at `0ccb34cb88b1cc5a127d1828f0072ca30133cc45`.
- #2 repository-control/protected-main alignment — partially complete. `main` is protected by active default-branch integrity and PR-review rulesets, but exact CUDA-family repository-setting parity is not yet complete.
- #3 consumer-backed algorithm activation roadmap — open and is the next semantic/implementation decision once #2 is fully aligned.

## Protected-main readback

`main` is protected by two active default-branch rulesets:

1. `Default branch integrity` blocks branch deletion and non-fast-forward updates, with no bypass actor.
2. `PR review - owner and ChatGPT exceptions` requires one approving review, dismisses stale reviews after pushes, requires review-thread resolution, requires extra approval for unattributed changes, and gives the repository owner plus the authorized ChatGPT integration PR-only bypass.

This matches the current sibling CUDA-family protection shape except for one remaining difference: sibling PR-review rulesets currently have code-owner review enabled, while CUDA-Algorithms has that flag disabled. No required status-check name is fabricated because CUDA-Algorithms has no local CI workflow yet.

## Repository-setting parity still open under #2

Current CUDA-Algorithms repository settings still differ from the established sibling baseline:

- auto-merge: **off**; sibling target **on**;
- merge commits: **on**; sibling target **off**;
- squash merge: **on**; matches;
- rebase merge: **on**; matches;
- update-branch support: **off**; sibling target **on**;
- delete merged head branches: **off**; sibling target **on**;
- web commit signoff required: **off**; sibling target **on**;
- wiki: **on**; sibling target **off**;
- discussions: **off**; sibling target **on**.

The connected GitHub integration can read repository/ruleset state but does not expose the repository-administration/ruleset mutation needed to close those gaps. #2 remains open until an authorized admin-capable path applies the remaining settings and the result is read back.

## Next executable decision

After #2 is fully aligned, select the smallest reusable consumer-backed algorithm profile under #3. Current evidence points toward a common primitive spine around scan, radix sorting, compaction, keyed uniquing/reduction and GPU-owned workset/closure execution, but the issue is planning authority rather than a production specification.

BSFP is the strongest immediate consumer for GPU-owned ranked closure, but CUDA-Algorithms must stay consumer-neutral. CUDA-DATA and CUDA-GRAPH-ANALYTICS provide materially different reuse tests for sequence/keyed and frontier/fixed-point primitives.

## Ecosystem boundaries

CUDA-JS owns generic Device-JS/compiler/runtime/memory/provider/lifecycle mechanisms. CUDA-JS-Tensor owns Tensor mathematics. CUDA-DATA owns tables/columns/dataframes. CUDA-GRAPH-ANALYTICS owns graph-analysis meaning. CUDA-MM owns generic physical placement/spill policy if activated. CUDA-MCGS and downstream solvers retain search/proof/domain semantics.

No bootstrap artifact, active ruleset, provider availability or roadmap item is production implementation/support/performance authority.
