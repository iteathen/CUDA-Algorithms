import test from 'node:test';
import assert from 'node:assert/strict';
import { stableLexicographicOrderIndices } from './permutation-ordering.mjs';
import {
  exactWideChangeAfter,
  groupIdsFromChangeAfter,
  independentWideGroupIds,
  representativeIndicesFromChangeAfter,
} from './grouping-boundary.mjs';

test('change-after plus exclusive scan yields exact group IDs for wide ordered records', () => {
  const words = [
    [2n, 1n, 2n, 1n, 2n, 1n],
    [0n, 9n, 0n, 8n, 0n, 8n],
    [5n, 1n, 4n, 7n, 4n, 6n],
  ];
  const order = stableLexicographicOrderIndices(words);
  const changes = exactWideChangeAfter(words, order);
  assert.deepEqual(groupIdsFromChangeAfter(changes), independentWideGroupIds(words, order));
});

test('grouping boundary survives 500 deterministic duplicate-heavy wide fixtures', () => {
  let state = 0x12345678n;
  const next = () => {
    state = (1664525n * state + 1013904223n) & 0xffffffffn;
    return state;
  };

  for (let fixture = 0; fixture < 500; fixture += 1) {
    const rows = 1 + Number(next() % 100n);
    const columns = 1 + Number(next() % 10n);
    const words = Array.from({ length: columns }, () =>
      Array.from({ length: rows }, () => next() % 29n));
    const order = stableLexicographicOrderIndices(words);
    const changes = exactWideChangeAfter(words, order);
    assert.deepEqual(
      groupIdsFromChangeAfter(changes),
      independentWideGroupIds(words, order),
      `fixture ${fixture}`,
    );
  }
});

test('consumer-owned data null equality can share the same generic segmentation mechanics', () => {
  const isNull = [false, true, false, true, false];
  const values = [2, 999, 2, 123, 3];

  // CUDA-DATA-like owner canonicalizes its own semantic key. Under this
  // fixture, all nulls compare equal regardless of irrelevant payload bytes.
  const canonicalNullRank = isNull.map((value) => value ? 0n : 1n);
  const canonicalValue = values.map((value, i) => isNull[i] ? 0n : BigInt(value));
  const order = stableLexicographicOrderIndices([canonicalNullRank, canonicalValue]);

  const changes = order.map((index, i) => {
    if (i + 1 >= order.length) return 0n;
    const a = Number(index);
    const b = Number(order[i + 1]);
    const equal = (isNull[a] && isNull[b]) ||
      (!isNull[a] && !isNull[b] && values[a] === values[b]);
    return equal ? 0n : 1n;
  });

  assert.deepEqual(
    groupIdsFromChangeAfter(changes),
    independentWideGroupIds([canonicalNullRank, canonicalValue], order),
  );

  const representatives = representativeIndicesFromChangeAfter(order, changes);
  assert.equal(representatives.length, 3);
});

test('invalid change-after shape fails closed', () => {
  assert.throws(() => groupIdsFromChangeAfter([0n, 1n]), /last changeAfter flag must be 0/);
  assert.throws(() => groupIdsFromChangeAfter([0n, 2n, 0n]), /must be 0 or 1/);
});
