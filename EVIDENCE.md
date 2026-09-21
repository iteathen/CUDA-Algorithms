# Evidence status

This repository follows the shared [iteathen evidence and validation policy](https://github.com/iteathen/.github/blob/main/EVIDENCE_POLICY.md).

## Current posture

CUDA-Algorithms is currently a planning/reference-semantics repository. It contains reference and candidate checks, but no production implementation, installable package, public API, native-provider qualification, or performance claim.

## Registered claims

| Claim | Evidence class | Status |
| --- | --- | --- |
| `CUDA-ALGORITHMS-PLAN-001` — intended scope: reusable provider-neutral GPU parallel algorithms in the CUDA-JS ecosystem | **UNVALIDATED** | planning hypothesis / project boundary |

The claim record is machine-readable in [`evidence/claims.json`](evidence/claims.json).

## What current evidence establishes

The repository establishes intended ownership boundaries and contains internal reference/candidate semantics useful for future implementation qualification.

## What it does not establish

It does not establish a production GPU algorithms library, native CUDA correctness, API stability, comparative performance, or external reproduction.

## Path to stronger evidence

When implementation is authorized, each algorithm should be checked against independently specified reference vectors or independently maintained implementations before native/hardware performance claims are promoted.

## Non-mutation rule

Evidence work may inspect, test, benchmark, and document CUDA-Algorithms. It must not change substantive algorithm implementation merely to make an evidence claim pass.
