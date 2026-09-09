# CUDA-Algorithms Status

**Updated:** 2026-09-09

**Architecture/ownership:** accepted independent reusable GPU parallel-algorithm semantic owner under ADR-0001 and SPEC-0001.
**Production implementation/API:** not authorized / none.
**Provider/support:** none selected or claimed.

## Current work

- #1 ownership/bootstrap authority — implementation of the documentation foundation is in progress on `bootstrap/open-source-foundation`.
- #2 repository-control/protected-main alignment — open. The target settings are defined, but the current GitHub integration does not expose repository-administration/ruleset mutation; protection must not be claimed until an authorized admin path applies and reads it back.
- #3 consumer-backed algorithm activation roadmap — open and is the next semantic/implementation decision after bootstrap/governance.

## Next executable decision

After #1 is integrated and #2 is completed through an admin-capable path, select the smallest reusable consumer-backed algorithm profile under #3. Current evidence points toward a common primitive spine around scan, radix sorting, compaction, keyed uniquing/reduction and GPU-owned workset/closure execution, but the issue is planning authority rather than a production specification.

BSFP is the strongest immediate consumer for GPU-owned ranked closure, but CUDA-Algorithms must stay consumer-neutral. CUDA-DATA and CUDA-GRAPH-ANALYTICS provide materially different reuse tests for sequence/keyed and frontier/fixed-point primitives.

## Ecosystem boundaries

CUDA-JS owns generic Device-JS/compiler/runtime/memory/provider/lifecycle mechanisms. CUDA-JS-Tensor owns Tensor mathematics. CUDA-DATA owns tables/columns/dataframes. CUDA-GRAPH-ANALYTICS owns graph-analysis meaning. CUDA-MM owns generic physical placement/spill policy if activated. CUDA-MCGS and downstream solvers retain search/proof/domain semantics.

## Governance

Target protected-main rules match the CUDA-family pattern: an active `Main` ruleset on the default branch blocks deletion and non-fast-forward updates; required checks remain empty until real local CI exists. Target repository settings also match the family: auto-merge on, merge commits off, squash/rebase on, update-branch on, delete merged branches on, web signoff required, wiki off, discussions on.

Do not mark #2 complete until those settings are actually mutated and read back.
