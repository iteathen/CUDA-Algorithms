# Native Qualification — First Algorithm Profile

**Status:** qualification harness; not production execution code
**Issue:** #3

## Purpose

Execute the current correctness-first Device-JS scan/select and permutation-ordering experiments on directly accessible CUDA hardware and compare physical GPU results against the independent JavaScript reference semantics.

This harness is deliberately separate from production execution. It may call `operation.wait()` and perform D2H reads because its job is to qualify results. The GPU-owned production design still forbids a synchronous Node semantic-advancement loop.

## Authority and package requirement

Run against the exact CUDA-JS revision currently used by the portable evidence:

```text
97c0295ab79add204d4d8ced080a4da4b66149cf
```

`cuda-js` must resolve through its **public package exports**. A local exact checkout may be linked/installed as the `cuda-js` package; do not change this harness to import CUDA-JS private repository files.

CUDA-JS's published hardware policy requires direct physical CUDA hardware for native evidence. Hosted CI, mocks, VMs and frontend inspection are not native qualification substitutes.

## Run

On a host satisfying the selected CUDA-JS native profile and supported Node version:

```text
node experiments/native-qualification/run.mjs
```

The harness currently exercises:

### Stable select / device-resident active extent

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

### Stable permutation ordering

- two-word stable lexicographic ordering from a non-identity incoming sequence;
- active prefix smaller than index capacity;
- duplicate index values / multiset preservation;
- zero active items;
- invalid indirect-index detection before key dereference.

For valid cases it compares final device indices against the independent permutation reference.

## Evidence output

The script emits one bounded JSON result containing fixture names/statuses and the final pass/fail outcome. It intentionally avoids printing hostnames, account paths, device UUIDs, PCI identifiers or other unnecessary host identity.

A passing run qualifies only the exact source/CUDA-JS/Node/GPU/driver/provider profile actually executed. Record those profile facts separately according to CUDA-JS hardware-evidence policy; do not infer broader support.

## Failure handling

The first failure must be treated as evidence. Do not weaken a fixture merely to obtain green output.

Classify whether the first divergence is:

- CUDA-Algorithms semantic/source defect;
- Working Draft design defect;
- CUDA-JS contract/frontend/runtime defect;
- unsupported/unqualified host profile;
- provider/native infrastructure failure;
- resource/lifecycle cleanup failure.

Repair the authoritative owner, then rerun the smallest affected qualification set.

## Cleanup

Every fixture closes its operation, prepared algorithm resources, views and allocations. The runtime closes once after the complete run. A non-graceful close or restart-required result fails qualification even if numerical outputs matched.
