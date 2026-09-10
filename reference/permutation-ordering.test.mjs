import test from 'node:test';
import assert from 'node:assert/strict';
import {
  independentLexicographicIndexOrder,
  stableLexicographicIndexOrder,
} from './core-primitives.mjs';
import {
  stableLexicographicOrderIndices,
  stableOrderIndicesByKey,
} from './permutation-ordering.mjs';

const B = (...values) => values.map(BigInt);

test('stable index ordering preserves current sequence order among equal keys', () => {
  assert.deepEqual(
    stableOrderIndicesByKey(B(9, 2, 9, 2), B(2, 3, 0, 1)),
    B(3, 1, 2, 0),
  );
});

test('permutation-first wide ordering equals pair-sort/gather composition and independent oracle', () => {
  const words = [
    B(2, 1, 2, 1, 2, 1),
    B(0, 9, 0, 8, 0, 8),
    B(5, 1, 4, 7, 4, 6),
  ];
  const indirect = stableLexicographicOrderIndices(words);
  assert.deepEqual(indirect, stableLexicographicIndexOrder(words));
  assert.deepEqual(indirect, independentLexicographicIndexOrder(words));
});

test('permutation-first wide ordering survives duplicate-heavy deterministic fixtures', () => {
  let state = 0x9e3779b9n;
  const next = () => {
    state = (1103515245n * state + 12345n) & 0xffffffffn;
    return state;
  };

  for (let fixture = 0; fixture < 250; fixture += 1) {
    const rows = 1 + Number(next() % 96n);
    const columns = 1 + Number(next() % 8n);
    const words = Array.from({ length: columns }, () =>
      Array.from({ length: rows }, () => next() % 23n));
    assert.deepEqual(
      stableLexicographicOrderIndices(words),
      independentLexicographicIndexOrder(words),
      `fixture ${fixture}`,
    );
  }
});

test('invalid indirect index rejects before key lookup', () => {
  assert.throws(
    () => stableOrderIndicesByKey(B(4, 5), B(0, 2)),
    /outside key storage/,
  );
});
