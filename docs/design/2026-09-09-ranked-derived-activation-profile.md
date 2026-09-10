# Ranked derived activation — first typed consumer composition profile

**Status:** working design/evidence for SPEC-0004; not Accepted compatibility authority  
**Date:** 2026-09-09  
**Issue:** #3

## Purpose

Record the smallest consumer-neutral GPU composition seam demonstrated by the first ranked-closure implementation work.

This profile is intentionally narrower than a generic callback system. It supports a finite implicit ranked item universe in which one active item can derive a bounded number of lower-rank target item indices.

## Deletion test

The public meaning is coherent with Connect4/BSFP deleted:

```text
active source item indices
  -> statically linked consumer derivation(sourceIndex, emissionLane)
  -> target item indices or INVALID
  -> strict generic rank validation
  -> duplicate-idempotent activation
  -> deterministic compact next workset
  -> device-resident next extent/status
```

Examples that fit without game vocabulary include:

- implicit dependency DAG activation;
- staged data-lineage invalidation/recomputation;
- acyclic dynamic-programming dependency scheduling;
- bounded recursive provenance graphs represented by indices.

The reference qualification includes both an implicit dependency-DAG consumer and an unrelated staged data-lineage consumer.

## Typed consumer seam

The first profile uses CUDA-JS SPEC-0028 typed Device-JS library composition.

The consumer supplies one compiled public leaf-library export with exact logical signature:

```text
(u32 sourceIndex, u32 emissionLane) -> u32 targetIndex
```

`0xffffffff` is the first-profile inactive-lane sentinel.

The signature is deliberately index-only. It does **not** define a generic consumer record ABI, opaque byte context, dynamic function pointer, proof callback, graph object, or domain predicate.

If a later real consumer demonstrates that index+lane cannot express the required reusable implicit derivation without duplicating generic lower work, expand the contract only from that evidence.

## Algorithm ownership

CUDA-Algorithms owns:

- host-known finite item/input/output capacities;
- finite `maxEmissionsPerItem`;
- active item-index workset meaning;
- strict rank-descent validation against consumer-supplied rank words;
- interpretation of the inactive-lane sentinel;
- duplicate-idempotent target activation;
- deterministic ascending-index compaction of the next active set in this first profile;
- device-resident next extent and algorithm status;
- prepared-epoch construction/lifecycle;
- output-capacity truth.

The consumer owns:

- what each item index means;
- why a target is a dependency of a source;
- rank meaning, subject to the declared strict ordering contract;
- domain records addressed by those indices;
- proof/value/equality/dominance semantics above simple target-index activation.

CUDA-JS owns Device-JS type checking, library/program identities, compile/link target compatibility, prepared execution, device views, operations, memory and native cleanup.

## First realization

The correctness-first Device-JS realization uses one prepared four-node epoch:

```text
reset activation flags/control
  -> derive + rank-check + CAS activation
  -> exact full-universe flag scan
  -> compact activated indices + next count
```

The full-universe scan is intentionally simple and may be slow. It is not a performance recommendation or claim. Its purpose is to expose the semantic/API seam using already accepted CUDA-JS mechanisms before lower optimization is justified.

No host count readback or host mathematical decision occurs between those four nodes.

## Capacity and failure truth

The first profile must fail/yield explicitly for:

- active extent greater than input capacity;
- source index outside the item universe;
- derived target outside the item universe;
- derived target whose rank does not strictly descend;
- next active set larger than output capacity.

A failed/capacity-limited epoch must not be treated as convergence.

## Relationship to BSFP

Connect4 CUDA-BSFP maps its support-skeleton predecessor relation onto this profile:

```text
support skeleton index
+ predecessor lane/column
  -> lower-rank support skeleton index or INVALID
```

That mapping does not transfer Connect Four support, CPC, WSL-625, NDC, W/D/L, existential/universal reduction, proof equality, or terminal semantics into CUDA-Algorithms.

In particular, ranked target activation is not itself BSFP value/proof evaluation. A BSFP predecessor becomes semantically authoritative only after Connect4-owned rank-complete contribution aggregation and exact existential/universal reduction.

## Evidence required before SPEC-0004 promotion

This profile alone is insufficient to promote general RankedClosure.

Still required:

- passing portable/package composition tests against the exact CUDA-JS pair;
- at least one real NVIDIA numerical run of the prepared epoch against the JavaScript reference;
- Connect4 BSFP consumer-side qualification of the support-lattice mapping;
- truthful capacity/rank failure observation on the physical path;
- cleanup evidence;
- evidence that the profile remains useful for at least one non-BSFP consumer;
- separate design/implementation for BSFP rank-complete proof/value reduction, which remains consumer-owned unless a truly generic lower algebra is independently demonstrated.

## Falsifiers

Retire or revise this profile if:

- typed leaf-library composition cannot execute the derivation safely through the public CUDA-JS surface;
- a correct consumer needs a generic record ABI merely to use it;
- physical shard/batch choice changes the activated target set;
- duplicate activations are not idempotent;
- rank or capacity errors can masquerade as a valid next workset;
- the only credible consumer is BSFP/Connect4.
