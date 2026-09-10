# Reference semantics

This directory contains deterministic JavaScript reference behavior used to define and falsify CUDA-Algorithms Working Draft semantics.

It is **not** a CPU production backend and does not authorize a production public API. GPU-owned consumers are not required to fall back to these functions at runtime.

## Current reference

`core-primitives.mjs` models the current SPEC-0002/SPEC-0003 draft subset with exact `BigInt` unsigned arithmetic rather than JavaScript Number bitwise coercion:

- inclusive/exclusive unsigned scan;
- unsigned reduction;
- stable flag-to-index selection;
- gather by index;
- stable primitive-word radix ordering semantics;
- run-length encoding;
- reduce-by-key;
- device-active-extent validation model;
- repeated stable word-pass composition and an independent lexicographic oracle for wide keys.

Run the current deterministic fixture suite with:

```text
node --test reference/core-primitives.test.mjs
```

The first recorded local qualification used Node v22.16.0 and passed 15/15 tests. That result proves only the reference implementation at its exact revision; it does not prove GPU/native support or performance.

## Draft discipline

These reference functions may change while their owning specifications remain `Working Draft`. Do not preserve prototype names or behavior for compatibility when implementation evidence exposes a better semantic contract.
