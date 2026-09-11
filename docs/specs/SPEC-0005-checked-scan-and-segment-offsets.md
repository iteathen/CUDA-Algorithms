# SPEC-0005 — Checked u32 scan and segment offsets

**Status:** Candidate; maintained implementation authorized by the owner's
explicit request to implement #9. Native qualification is required before any
native support claim. Depends on SPEC-0001 and Candidate SPEC-0002. Existing
SPEC-0003 and modular reference sums are unchanged.

## Purpose and ownership

Provide checked exclusive addition and its bounded segment/representative-offset
composition. CUDA-Algorithms owns arithmetic, stability, capacities and sequence
mechanics; consumers own ordering, exact equality, group boundaries and length
units. Deleting OQS leaves ordinary row-group and sequence-offset consumers.
Neither operation copies payloads, sorts records or discovers equality.

## Public plans

`createCheckedExclusiveScanU32Plan(runtime, { inputCapacity, blockSize?, maxWorkspaceBytes? })`
binds `input: u32[inputCapacity] read`, `activeCount: u32[1] read`,
`upstreamStatus: u32[1] read`, `prefix: u32[inputCapacity] write`,
`total: u32[1] write`, `outputCount: u32[1] write`, `status: u32[1] read-write`.

On success, `prefix[i] = sum(input[0..i))`, `total = sum(input[0..activeCount))`,
and `outputCount = activeCount`. These sums are mathematical nonnegative integer
sums, not modular arithmetic. UINT32_MAX is valid; any larger total fails.
Empty input yields zero total/count. Inactive tails have no semantic effect.

`createSegmentOffsetsU32Plan(runtime, { inputCapacity, groupCapacity, blockSize?, maxWorkspaceBytes? })`
binds `heads: u32[inputCapacity] read`, `lengths: u32[inputCapacity] read`,
`activeCount: u32[1] read`, `upstreamStatus: u32[1] read`,
`groupIds: u32[inputCapacity] write`,
`representatives`, `compactLengths`, `offsets`: each `u32[max(1,groupCapacity)] write`,
and `groupCount`, `requiredGroups`, `totalLength`: each `u32[1] write`,
plus `status: u32[1] read-write`.

Active head flags must be binary, with first flag one for nonempty input. Each
head starts a segment. Group IDs are dense in input order; representatives are
the stable input positions of heads. Compact length is the length at the head;
other member lengths are irrelevant. Offsets are exclusive checked sums of
compact lengths. Length zero is valid. `groupCapacity=0` admits empty input only.
Caller-established segments need not have equal lengths or any library-owned
record identity. `requiredGroups` retains the exact needed count after valid
heads, including on group-capacity failure. No failed payload is valid.

## Failure, composition and concurrency

Shared status vocabulary: `OK=0`, `INVALID_EXTENT=1`, `INVALID_HEAD=2`,
`SUM_OVERFLOW=3`, `GROUP_CAPACITY_EXHAUSTED=4`, `UPSTREAM_FAILED=5`.
Upstream nonzero is translated to UPSTREAM_FAILED before any live input read;
the original upstream word is preserved. Extent failure precedes head validation;
valid heads precede capacity checks; length overflow follows representative
selection. Success counts/totals are zero on failure; requiredGroups may remain.
Payloads, including workspace, are unspecified on failure. Status must be checked
or forwarded on device; a zero output count alone never proves success.

`submit(bindings, { after? })` returns the ordinary CUDA-JS operation. `after`
passes through to public prepared execution. A different plan may consume device
outputs/counts/status using this explicit dependency without any host data read.
Overlapping two-operation chaining requires the public CUDA-JS execution profile
`maxPendingGpuOperations: 2`; capacity-one runtimes may submit after terminality.
Inputs and output bindings remain caller-owned and leased by CUDA-JS execution.

Each plan owns finite disjoint workspace, module/functions and its prepared DAG.
Overlapping submissions and close during submission/pending execution reject;
caller waits for terminality before reusing or closing that plan. Different plans
have independent workspace. Runtime failure/orphan state is never reported as
algorithm success. Construction failures release all already-created resources;
cleanup failures propagate. Caller owns returned-operation close.

All public roles validate u32 dtype, access and minimum capacity. All write roles
must be disjoint from other roles; overlapping read-only roles are allowed. Public
CUDA-JS range inspection rejects stale/cross-runtime capabilities and determines
range truth. No private allocation registry or native address is introduced.

## Finite realization

Current realization admits inputCapacity 1..262144 and blocks 64/128/256; these
are qualified profile limits, not mathematical u32 limits. groupCapacity is an
integer 0..inputCapacity. `checkedScanU32Requirements(options)` and
`segmentOffsetsU32Requirements(options)` return deterministic workspace bytes,
external binding element/access requirements and prepared-node count before any
allocation. `maxWorkspaceBytes`, when supplied, rejects insufficient workspace
budget before compilation/allocation. Runtime overhead and caller-owned buffers
are additional to workspace bytes and remain caller admission responsibilities.

Hierarchical block scans use disjoint global ping-pong scratch and public block
barriers. Work O(N log B), scratch O(N), nodes O(log_B N), at most 32. No cross-block
spin, data-dependent host progression or hidden allocation during submit. Every
lane reaches every block barrier; mutable status never controls divergent exit.
Ordered guard setup supplies a uniform safe active extent to the scan stages.

## Qualification

Retain independent BigInt expected results, OQS and nullable-row fixtures, empty
and singleton extents, partial tiles, all-equal/all-distinct groups, zero lengths,
exact UINT32_MAX, within/across-tile overflow, invalid flags/extents, output
capacity, recovery, block invariance, upstream failure and explicit device
count/status chaining. Verify public alias/access/stale/cross-runtime rejection,
in-flight/closed lifecycle and injected partial-construction/cleanup failures.
Benchmark stable representatives against existing quadratic selection on safe
bounded head flags; measure the same emitted indices, reporting differing extra
work explicitly. Portable evidence is not numerical/native evidence.
