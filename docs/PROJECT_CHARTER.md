# CUDA-Algorithms Project Charter

**Status:** Accepted architecture/ownership authority; production implementation not authorized.

## Purpose

CUDA-Algorithms exists to own reusable, provider-neutral GPU parallel-algorithm semantics that are genuinely shared across materially different CUDA-JS ecosystem consumers, without turning CUDA-JS into an algorithms framework or moving domain meaning out of its natural owner.

## Owns, when separately accepted

- prefix scan and segmented scan semantics;
- reductions and keyed/segmented reductions;
- selection, filtering support, stream compaction and partition primitives;
- radix/key sorting and sort-pair semantics;
- run-length encoding, unique-by-key and reduce-by-key;
- gather/scatter and related index-driven movement semantics where they are algorithmic rather than raw transfer mechanics;
- bounded histogram/bucketing primitives where provider-neutral semantics are established;
- generic frontier/workset representations and progression;
- ranked acyclic closure and general monotone fixed-point/workset execution semantics;
- deterministic algorithm plans, scratch/resource requirements, stability/determinism policy and provider-neutral conformance.

## Does not own

- Device-JS language/compiler/runtime, native CUDA/provider mechanisms, allocations, streams, transfers, atomics, operation lifecycle or native cleanup — CUDA-JS owns those;
- Tensor dtype/shape/layout/math semantics — CUDA-JS-Tensor;
- table/column/schema/join/group-by/dataframe meaning — CUDA-DATA;
- graph representation, BFS/SSSP/components/PageRank or other graph-analysis meaning — CUDA-GRAPH-ANALYTICS;
- search/MCGS semantics — CUDA-MCGS;
- CPC, WSL-625, NDC, BSFP, game rules or other domain-proof semantics — their downstream owner;
- generic physical placement, spill/eviction/migration/prefetch policy — CUDA-MM when activated;
- filesystem/source/sink semantics — cuda-io;
- communication/collective semantics — cuda-comm;
- arbitrary vendor-library passthrough or native provider APIs.

## Composition

CUDA-Algorithms depends downward on public CUDA-JS contracts for realization. It may consume CUDA-MM or cuda-io only through separately accepted optional profiles; neither becomes semantic authority for the algorithms themselves.

Higher libraries and applications consume CUDA-Algorithms without transferring their own domain semantics into it. A generic `RankedClosure` may execute BSFP work, for example, but BSFP's proof algebra remains outside this repository.

## Implementation rule

Maintained implementation/reference source is JavaScript/TypeScript. Restricted Device-JS is allowed only through public CUDA-JS contracts under accepted profiles. No repository-local C/C++/CUDA/PTX/native FFI/provider binding or copied CUDA ABI/native authority is permitted.

## Activation gate

Issue #3 must select a bounded consumer-backed profile and prove that its semantics survive deletion of any one first consumer. Reference semantics, finite resource bounds, falsifiers, qualification requirements and an accepted specification are required before production source/API implementation.

## Deletion/substitution rule

Deleting CUDA-Algorithms leaves CUDA-JS complete and leaves every upper consumer semantically complete; consumers may fall back to owner-local reference algorithms or another qualified implementation. Deleting BSFP, CUDA-DATA, CUDA-GRAPH-ANALYTICS or any one provider leaves CUDA-Algorithms coherent.

## Non-goals

Universal STL replacement, BSFP-specific accelerator, graph library, dataframe library, tensor library, hidden global scheduler, physical memory manager, arbitrary CUDA/CUB passthrough, or implementation merely because CUDA exposes a primitive.
