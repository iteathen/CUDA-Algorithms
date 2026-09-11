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
