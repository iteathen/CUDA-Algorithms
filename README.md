# CUDA-Algorithms

CUDA-Algorithms is a planned JavaScript library for reusable provider-neutral GPU parallel algorithms in the CUDA-JS ecosystem.

## Current state

This repository currently contains the project charter, architecture decision, development guidance, and planning records. **There is no production implementation, installable package, or public API yet.** No native-provider support or performance is claimed.

## Intended scope

The library is intended to own reusable algorithm semantics such as scan, reduction, selection/compaction, partitioning, sorting, keyed reduction/uniquing, gather/scatter, and bounded GPU-resident frontier/workset/fixed-point execution.

CUDA-JS supplies generic GPU/runtime/compiler/memory mechanisms. CUDA-JS-Tensor owns tensor mathematics; CUDA-DATA owns column/table/dataframe semantics; CUDA-GRAPH-ANALYTICS owns graph-analysis meaning; CUDA-MM owns reusable physical memory-management policy; domain solvers such as BSFP retain their own mathematical semantics.

Implementation depends on a concrete consumer need and an accepted specification. The activation roadmap will select the smallest reusable profile justified by materially different consumers rather than turning the library into a BSFP-specific package.

## Start here

- [Current status](STATUS.md).
- [Project charter](docs/PROJECT_CHARTER.md) and [documentation](docs/README.md).
- [Development instructions](AGENTS.md) and [shared contribution guide](https://github.com/iteathen/.github/blob/main/CONTRIBUTING.md).
- [Private security reporting](https://github.com/iteathen/.github/blob/main/SECURITY.md).
- [License](LICENSE).
