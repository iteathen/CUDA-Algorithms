# Native Qualification — First Algorithm Profile

**Status:** qualification harness; not production execution code
**Issue:** #3

## Purpose

Execute the maintained correctness-first CUDA-Algorithms Candidate plans on directly accessible CUDA hardware and compare physical GPU results against independent JavaScript reference semantics.

This harness is deliberately separate from production execution. It may call `operation.wait()` and perform D2H reads because its job is to qualify results. The GPU-owned production design still forbids a synchronous Node semantic-advancement loop.

## Authority and package requirement

Run against the exact CUDA-JS revision currently used by Candidate portable evidence:

```text
98e2ebc942c14d63acf4dd82e912dd548c363a05
cuda-js@0.1.0-alpha.20
Node v26.7.0
```

CUDA-JS's earlier gate-32 evidence proves its recorded alpha.19 Windows x64 / GTX 1660 Ti / driver 610.74 / CUDA 13.3 / Node 26.7.0 compatible pair. Alpha.20 adds the public pure `inspectDeviceViewRelation` facade stud and remains the exact lower revision CUDA-Algorithms now consumes.

Do not silently treat the old alpha.19 package evidence as alpha.20 qualification. The CUDA-Algorithms physical run must record the exact alpha.20 source/package/profile it actually exercises.

`cuda-js` must resolve through its **public package exports**. A local exact checkout may be linked/installed as the `cuda-js` package; do not import CUDA-JS private repository files.

## Run

On the directly accessible Windows CUDA host with the exact lower revision bound as the public package:

```text
node experiments/native-qualification/run.mjs
```

The harness imports the maintained `src/` Candidate surface rather than neighboring prototype plans.

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

Key-word views are supplied most-significant to least-significant. The Candidate plan performs stable passes in reverse significance and reports `resultBinding`, identifying which ping-pong index view contains the final permutation.

For valid cases it compares final device indices against `reference/permutation-ordering.mjs`.

## Alias boundary exercised before native launch

The maintained Candidate consumes CUDA-JS public `inspectDeviceViewRelation` and requires:

```text
read + read: overlap may be permitted
any pair involving write: relation must be disjoint
```

Same-range or partially overlapping write-conflicting views fail before algorithm submission. CUDA-Algorithms owns that policy; CUDA-JS owns the byte-range fact.

## Evidence output

The script emits one bounded JSON result containing fixture names/statuses and the final pass/fail outcome. It intentionally avoids printing hostnames, account paths, device UUIDs, PCI identifiers or other unnecessary host identity.

A passing run qualifies only the exact CUDA-Algorithms source / CUDA-JS / Node / GPU / driver / provider profile actually executed. Do not infer broader support.

## Failure handling

The first failure is evidence. Do not weaken a fixture merely to obtain green output.

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
