# ADR-0001: Independent GPU Parallel-Algorithm Semantic Owner

**Status:** Accepted
**Date:** 2026-09-09

## Context

Multiple CUDA-JS ecosystem consumers need reusable GPU data-parallel and irregular-work algorithms such as scan, sort, compaction, keyed reduction, frontier progression and fixed-point closure. Those semantics are not native CUDA runtime vocabulary, Tensor mathematics, table/dataframe meaning, graph-analysis meaning, search semantics or physical memory-management policy.

Putting them in CUDA-JS would turn the runtime/compiler layer into an algorithms framework. Putting them in CUDA-JS-Tensor would incorrectly require tensor semantics. Putting them in CUDA-DATA or CUDA-GRAPH-ANALYTICS would make one domain library the hidden algorithm core for unrelated consumers such as BSFP.

## Decision

`CUDA-Algorithms` is the independent owner for reusable provider-neutral GPU parallel-algorithm semantics and plans.

CUDA-JS retains generic GPU/compiler/runtime/memory/provider/lifecycle mechanisms. CUDA-JS-Tensor retains Tensor mathematics. CUDA-DATA, CUDA-GRAPH-ANALYTICS, CUDA-MCGS, BSFP and other consumers retain their domain semantics. CUDA-MM remains the owner for reusable physical memory-management policy when activated.

## Initial consumer evidence

The ownership boundary is already motivated by materially different consumers:

- Connect4 BSFP/NDC needs scan, sort, unique/reduce-by-key, compaction, frontier/workset progression and ranked closure while retaining CPC/WSL-625/NDC/BSFP semantics downstream;
- CUDA-DATA needs selection/compaction, sorting/partitioning and keyed reduction while retaining table/column semantics;
- CUDA-GRAPH-ANALYTICS needs frontier/workset and fixed-point machinery while retaining graph representations and named graph algorithms.

These are evidence for a reusable layer, not authorization to ship every candidate primitive.

## Deletion test

Deleting any one consumer/provider leaves CUDA-Algorithms coherent. Deleting CUDA-Algorithms leaves CUDA-JS and each semantic consumer coherent; a consumer may substitute its own reference implementation without changing its domain meaning.

## Implementation gate

Issue #3 selects the first bounded consumer-backed algorithm profile. An accepted specification, independent/reference semantics, finite work/resource bounds and qualification plan are required before production source/API.

## Consequences

The ecosystem gains one visible owner for reusable GPU algorithms without conflating mechanisms with algorithms or turning a first consumer into the shared substrate.
