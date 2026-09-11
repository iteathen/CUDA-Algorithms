# Bounded segment scan reconnaissance

Producer-owned disposable experiment from CUDA-Algorithms
48ee0aec9acae7776950f03ab52ab1737e598b6e, motivated by Connect4 O3 at
5dfe1312a357c48eee53168e82fd6eba27814a06 and issue #3. This is not a new public
API, Candidate promotion, or production implementation. Existing exports/pins
are protected. The accepted ADR-0001 production gate still applies.

Question: can public Device-JS express a bounded nonquadratic scan that gives
dense segment IDs, stable representatives, compact representative lengths and
exclusive record offsets in one prepared submission, with device-owned counts?
Consumers supply already ordered items and exact binary segment-head flags.
No sorting, hash identity, record equality or domain semantics are implemented.

Inputs: head flags and u32 representative lengths, active extent <= capacity.
The first live flag must be one. Outputs: one dense group ID per input position,
stable representative positions, compact lengths, offsets, group count and total
length. Empty input produces zero counts. Group capacity exhaustion, invalid
extent/flags and u32 total overflow are explicit failures; partial outputs are
not valid. Length zero and exact UINT32_MAX total are legal. No payload copy is
claimed. Equal-group member lengths are irrelevant except at the head.

Realization: each block performs ping-pong inclusive addition in disjoint global
scratch, with public block barriers between reads and writes. Tile totals recurse
to one block, then prefix carries propagate downward. Work is O(N log B), scratch
O(N), prepared nodes O(log_B N). No cross-block spin or host progression. Addition
checks overflow before wrapping. Every lane reaches every barrier; live status
never controls divergent early return inside a barrier-bearing kernel.

Finite experiment envelope: capacity 1..262144; blocks 64/128/256; at most 32 DAG
nodes. All arrays are owned by the experiment plan, so aliasing is disjoint by
construction. Allocation admission is the exact simultaneously live array sum
plus 256 MiB runtime allowance. Native supervision uses a 30-second child timeout
and current free-VRAM admission. Retain failed raw evidence; success requires
reverse cleanup and graceful close.

Sequential plan: implement the bounded device prototype; compare independent
BigInt references on sequence and nullable-row controls, extent/tile boundaries,
overflow/capacity/failure-reuse and multiple block sizes; measure a short scaling
ladder; review and record the producer capability disposition. Rollback means
discard this isolated experiment, never modify the original dependency worktree.
Full OQS grouping and production scan API acceptance remain separate gates.

## Native outcome

Exact native source: `7d923eb5e4bf4664feab0d0d7c3dded2f81e15b3`.
[Raw result and preflight](../../docs/evidence/2026-09-11-segment-scan-native.json)
retain the exact source/lower/Node/GPU/driver identity and raw-log hashes.
All 72 fixtures / 84 submissions passed inside the original 30-second child
limit, including all three block sizes, inactive tails, checked overflow,
capacity errors and valid reuse. The OQS fixture produces 48 groups and offsets
for 10,597 records. The nullable-row fixture is a separate consumer shape.
Both supply CPU-prepared boundary flags; device equality is not tested.

| Input entries | Prepared nodes | Device bytes | Measured median submit/wait (ms) |
| ---: | ---: | ---: | ---: |
| 128 | 5 | 5,660 | 0.4901 |
| 8,192 | 9 | 364,060 | 0.6262 |
| 65,536 | 13 | 2,903,100 | 0.8847 |
| 262,144 | 13 | 11,603,100 | 0.9921 |

Each timing uses one warmup and three measured repetitions. All samples and
separate upload/readback/setup times are retained. Overhead dominates these short
submissions; do not infer linear extrapolation, pure kernel throughput or an
end-to-end solver speedup. No old-quadratic-path comparison was run.

All 44 local reference/Candidate/experiment tests and document validation pass.
[Portable/reference CI 34565710334](https://github.com/iteathen/CUDA-Algorithms/actions/runs/34565710334)
also passes at the exact native source revision.
Author-side bounded review checked ping-pong read/write separation, uniform
barrier reach, hierarchy tail initialization, u32 product/range limits,
failure-before-result publication and reverse resource disposal. This is not
independent acceptance of a public API.

The supported-library gap is filed as
[CUDA-Algorithms #9](https://github.com/iteathen/CUDA-Algorithms/issues/9), beneath
the existing #3 activation issue. No CUDA-JS mechanism gap was demonstrated.
Next: specify and qualify the checked scan/select public composition, preserving
the distinction from modular sums and keeping semantic equality with consumers.
Then qualify a deliberately pinned Connect4 composition. Scalable ordering and
payload copying remain separate unfinished work.

Retain this isolated research worktree/branch, its ignored CUDA-JS junction and
native/portable spools. The original dependency worktree, public source/exports,
accepted specifications and Connect4 dependency pins are unchanged. No native
process or device allocation is intentionally retained. No protected-main merge
or package/API promotion was performed.
