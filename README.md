# CUDA-Algorithms

CUDA-Algorithms is a planned JavaScript library for reusable provider-neutral GPU parallel algorithms in the CUDA-JS ecosystem.

## Current state

This repository currently contains the project charter, architecture decision, development guidance, and planning records. **There is no production implementation, installable package, or public API yet.** No native-provider support or performance is claimed.

## Intended scope

The library is intended to own reusable algorithm semantics such as scan, reduction, selection/compaction, partitioning, sorting, keyed reduction/uniquing, gather/scatter, and bounded GPU-resident frontier/workset/fixed-point execution.

CUDA-JS supplies generic GPU/runtime/compiler/memory mechanisms. CUDA-JS-Tensor owns tensor mathematics; CUDA-DATA owns column/table/dataframe semantics; CUDA-GRAPH-ANALYTICS owns graph-analysis meaning; CUDA-MM owns reusable physical memory-management policy; domain solvers such as BSFP retain their own mathematical semantics.

Implementation depends on a concrete consumer need and an accepted specification. The activation roadmap will select the smallest reusable profile justified by materially different consumers rather than turning the library into a BSFP-specific package.

## Start here

Candidate checked integer APIs are available under
[SPEC-0005](docs/specs/SPEC-0005-checked-scan-and-segment-offsets.md):

```js
import { checkedScanU32Requirements, createCheckedExclusiveScanU32Plan,
  segmentOffsetsU32Requirements, createSegmentOffsetsU32Plan } from 'cuda-algorithms';

const options = { inputCapacity: 8192 };
const requirements = checkedScanU32Requirements(options); // before allocating
const plan = await createCheckedExclusiveScanU32Plan(runtime, options);
const operation = await plan.submit(bindings, { after: precedingOperation });
await operation.wait(); // qualification/administration; no input/result math
await operation.close();
await plan.close();
```

Bindings and external buffer sizes/accesses are listed in `requirements.bindings`.
`workspaceBytes` is plan-owned device memory; caller buffers/runtime overhead are
additional. `upstreamStatus` is required (zero for the first stage). Checked sums
reject overflow; they do not change existing wrapping reference semantics.
Segment plans consume caller-owned exact group boundaries and lengths; they do
not sort, decide equality or copy payloads. The maintained API passed native
qualification on GTX 1660 Ti through CUDA-JS 98e2ebc / Node 26.7.0; see
[exact evidence](docs/evidence/2026-09-11-checked-scan-api-native.json). Candidate
status retains a breakable pre-acceptance contract, not a stable release promise.

- [Current status](STATUS.md).
- [Project charter](docs/PROJECT_CHARTER.md) and [documentation](docs/README.md).
- [Development instructions](AGENTS.md) and [shared contribution guide](https://github.com/iteathen/.github/blob/main/CONTRIBUTING.md).
- [Private security reporting](https://github.com/iteathen/.github/blob/main/SECURITY.md).
- [License](LICENSE).
