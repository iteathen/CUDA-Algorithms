# Repository context: CUDA-Algorithms

Universal engineering and design guidance comes from the account-global `AGENTS.md`.

## Mission and ownership

CUDA-Algorithms owns reusable provider-neutral GPU parallel-algorithm semantics when accepted: scans/reductions, selection/compaction, partitioning, sorting, keyed grouping/uniquing/reduction, gather/scatter, histogram-like primitives, and bounded frontier/workset/fixed-point execution plans.

CUDA-JS owns generic GPU/runtime/compiler/memory/provider/lifecycle mechanisms. CUDA-JS-Tensor owns generic Tensor mathematics. CUDA-DATA owns column/table/dataframe semantics. CUDA-GRAPH-ANALYTICS owns graph-analysis meaning. CUDA-MM owns reusable physical memory-management policy. CUDA-MCGS/search engines and domain solvers such as BSFP retain their own search/proof/domain semantics.

## Local routing

Accepted `docs/decisions/`, `docs/specs/`, repository status/roadmap, and current issues own local implementation/activation truth.

## Local constraints

Maintained implementation uses JavaScript/ESM plus accepted Device-JS through public lower contracts; no Python, repository-local C/C++/CUDA/PTX/native FFI, provider escape path, or private lower-repository imports. Missing generic GPU mechanisms route to CUDA-JS instead of being copied locally.
