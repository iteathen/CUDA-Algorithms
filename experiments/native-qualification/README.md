# Native Qualification — First Algorithm Profile

**Status:** qualification harness; not production execution code
**Issue:** #3

## Purpose

Execute the current maintained correctness-first CUDA-Algorithms candidate plans on directly accessible CUDA hardware and compare physical GPU results against the independent JavaScript reference semantics.

This harness is deliberately separate from production execution. It may call `operation.wait()` and perform D2H reads because its job is to qualify results. The GPU-owned production design still forbids a synchronous Node semantic-advancement loop.

## Authority and package requirement

Run against the current gate-32-approved CUDA-JS revision used by portable candidate evidence:

```text
e9837f20acf7901d445a1e7a2045459a1ae0118a
cuda-js@0.1.0-alpha.19
```

CUDA-JS records the exact Windows x64 / GTX 1660 Ti / driver 610.74 / CUDA 13.3 / Node 26.7.0 physical compatible pair as passed, reviewed and owner-approved at that revision. That lower evidence establishes that the required public CUDA-JS runtime/compiler/execution substrate is physically available on the recorded host profile; it does **not** automatically qualify CUDA-Algorithms results.

`cuda-js` must resolve through its **public package exports**. A local exact checkout may be linked/installed as the `cuda-js` package; do not change this harness to import CUDA-JS private repository files.

## Run

On a host satisfying the selected CUDA-JS native profile and supported Node version:

```text
node experiments/native-qualification/run.mjs
```

The harness imports the maintained `src/` candidate surface rather than the neighboring prototype plans.

### Stable select / device-resident active extent

Exercises:

- sparse selection;
- active prefix smaller than capacity;
- zero active items within positive physical capacity;
- output capacity exhaustion with required-count reporting and no success-shaped truncated result;
- invalid flag detection;
- invalid active extent.

For valid cases it compares:

```text
semantic status
output count
exclusive prefix values
stable selected indices
```

against `reference/core-primitives.mjs`.

### Stable lexicographic permutation ordering

Exercises:

- two-word stable lexicographic ordering from a non-identity incoming sequence;
- three-word ordering through the maintained generalized key-word plan;
- active prefix smaller than index capacity;
- duplicate index values / multiset preservation;
- zero active items;
- invalid indirect-index detection before key dereference.

Key-word views are supplied most-significant to least-significant. The candidate plan performs stable passes in reverse significance and reports `resultBinding`, identifying which ping-pong index view contains the final permutation.

For valid cases it compares final device indices against `reference/permutation-ordering.mjs`.

## Current alias boundary

The maintained candidate rejects exact same-view conflicts whenever either logical role writes. Pure read/read same-view reuse remains legal.

Distinct sibling views that overlap one underlying allocation cannot yet be classified from the current public CUDA-JS view surface. That consumer-neutral lower relation is tracked as `iteathen/CUDA-JS#260`. A native pass must therefore not be promoted into a claim that every possible overlapping sibling-view conflict is already detected.

## Evidence output

The script emits one bounded JSON result containing fixture names/statuses and the final pass/fail outcome. It intentionally avoids printing hostnames, account paths, device UUIDs, PCI identifiers or other unnecessary host identity.

A passing run qualifies only the exact CUDA-Algorithms source / CUDA-JS / Node / GPU / driver / provider profile actually executed. Record those profile facts separately according to CUDA-JS hardware-evidence policy; do not infer broader support.

## Failure handling

The first failure must be treated as evidence. Do not weaken a fixture merely to obtain green output.

Classify whether the first divergence is:

- CUDA-Algorithms semantic/source defect;
- Candidate/Working Draft design defect;
- CUDA-JS contract/frontend/runtime defect;
- unsupported/unqualified host profile;
- provider/native infrastructure failure;
- resource/lifecycle cleanup failure.

Repair the authoritative owner, then rerun the smallest affected qualification set.

## Cleanup

Every fixture closes its operation, maintained algorithm plan resources, views and allocations. The runtime closes once after the complete run. A non-graceful close or restart-required result fails qualification even if numerical outputs matched.
