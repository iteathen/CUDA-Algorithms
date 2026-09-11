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
