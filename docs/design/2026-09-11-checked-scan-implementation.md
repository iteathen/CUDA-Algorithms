# Checked-scan implementation assessment

Owner request: implement CUDA-Algorithms #9. Base 3aa4f6e, with native prototype
7d923eb and unchanged CUDA-JS 98e2ebc. Work is isolated on
`codex/checked-scan-issue-9`; original dependency worktrees/pins are protected.

One integration owner; sequential units: define Candidate SPEC-0005 and lift the
qualified hierarchy into an internal producer module; expose standalone checked
scan and segment-offset plans with public bindings/status/lifecycle; qualify API
falsifiers and native consumers/chaining/old-selection comparison; publish exact
evidence and a reviewable PR. Preserve reserve for failure diagnosis and cleanup;
defer ordering, payload copying and consumer migration rather than weakening
qualification. No independent-review claim or automatic main merge.

The prototype's all-owned fixture runner is not the public API. Keep data upload,
readback, reference equality and qualification timing in tests/experiments. The
library owns only workspace and prepared execution. Requirements are available
before allocation. Explicit upstream status avoids accepting failed zero counts.
Public operation dependencies permit GPU-produced count/status chaining.

Rejected: exporting the experimental fixture runner, changing wrapping sums,
requiring per-level caller buffers, generic record equality callbacks, or a new
competing operation lifecycle. Use existing CUDA-JS operations/range authority.
Two consumers remain coherent without Connect4. Existing library exports remain
compatible; no old quadratic implementation changes until measured comparison.

Falsifiers: wrong prefix/group/offset, overflow accepted, failed dependency becoming
empty success, invalid range dereference, divergence around barriers, overlapping
workspace execution, hidden host semantic decision or false cleanup. Freeze the
native source after API tests; preserve failed runs and repair the owning boundary.

## Outcome and disposition

Implemented both public Candidate APIs; native source f82997ae passes the full
bounded qualification and CI. The internal hierarchy is shared, with owned
workspace and caller-owned public bindings. The standalone output is published
only after checked total validity; segment counts/totals remain zero on error.
Prior exact required-group counts are preserved separately. Explicit upstream
status prevents failed counts from looking like successful empty work.

Public surface review checked every input/output access role against all DAG
uses, external-versus-private disjointness, uniform guards/barriers, finite
hierarchy indices, checked sums and terminal-state workspace reuse. Tests cover
construction and cleanup failures plus rejection during submission and after
close. Review is author-side; no independent review is implied.

Two integration findings were resolved: prepared binding identifiers must start
with a letter, and overlapping two-operation chaining requires the existing
CUDA-JS maxPendingGpuOperations=2 profile. Neither is a missing lower capability.
The test harness now waits for submitted predecessors even when dependent
submission fails, preserving cleanup and the primary failure.

Native run f82997ae retains every sample and separates upload/readback from
submit/wait. Median old/new selection comparison is 13.8351/13.6546 ms with large
variation, so no clear speedup is claimed. The new path computes additional group
IDs/compact lengths/checked offsets. Work remains nonquadratic by construction;
that asymptotic fact is separate from measured end-to-end latency.

Retain native/portable spools and the isolated worktree/package junction for
continuation; publish the exact final evidence and reviewable branch. The initial
successful API run at 3da32c2 is retained locally; f82997ae adds direct large
standalone-prefix checks and supersedes it for the final qualification claim.
No original dependency checkout, lower pin, protected trunk or consumer was
changed. Issue closure follows integration; no automatic merge is performed.
