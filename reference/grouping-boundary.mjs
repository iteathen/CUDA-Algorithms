import { scanUnsigned, selectIndices, gather } from './core-primitives.mjs';

function bit(value, label) {
  const v = typeof value === 'bigint' ? value : BigInt(value);
  if (v !== 0n && v !== 1n) throw new RangeError(`${label} must be 0 or 1`);
  return v;
}

export function groupIdsFromChangeAfter(changeAfter) {
  const flags = changeAfter.map((value, i) => bit(value, `changeAfter[${i}]`));
  if (flags.length > 0 && flags.at(-1) !== 0n) {
    throw new RangeError('last changeAfter flag must be 0');
  }
  return scanUnsigned(flags, { width: 64, mode: 'exclusive', op: 'add', init: 0n });
}

export function representativePositionsFromChangeAfter(changeAfter) {
  const flags = changeAfter.map((value, i) => bit(value, `changeAfter[${i}]`));
  if (flags.length === 0) return [];
  if (flags.at(-1) !== 0n) throw new RangeError('last changeAfter flag must be 0');

  const starts = new Array(flags.length).fill(0n);
  starts[0] = 1n;
  for (let i = 1; i < starts.length; i += 1) starts[i] = flags[i - 1];
  return selectIndices(starts, { indexWidth: 64 }).indices;
}

export function representativeIndicesFromChangeAfter(orderedIndices, changeAfter) {
  if (orderedIndices.length !== changeAfter.length) throw new RangeError('orderedIndices/changeAfter lengths differ');
  return gather(orderedIndices, representativePositionsFromChangeAfter(changeAfter));
}

export function exactWideChangeAfter(wordColumns, orderedIndices) {
  if (!Array.isArray(wordColumns) || wordColumns.length === 0) throw new TypeError('wordColumns must be non-empty');
  const n = wordColumns[0].length;
  if (wordColumns.some((column) => column.length !== n)) throw new RangeError('word columns must have equal length');

  const order = orderedIndices.map((value, i) => {
    const v = typeof value === 'bigint' ? value : BigInt(value);
    if (v < 0n || v >= BigInt(n)) throw new RangeError(`orderedIndices[${i}] outside record storage`);
    return Number(v);
  });

  return order.map((record, i) => {
    if (i + 1 >= order.length) return 0n;
    const next = order[i + 1];
    return wordColumns.some((column) => BigInt(column[record]) !== BigInt(column[next])) ? 1n : 0n;
  });
}

export function independentWideGroupIds(wordColumns, orderedIndices) {
  if (orderedIndices.length === 0) return [];
  const n = wordColumns[0].length;
  const order = orderedIndices.map((value) => Number(value));
  let group = 0n;
  const ids = [group];
  for (let i = 1; i < order.length; i += 1) {
    const previous = order[i - 1];
    const current = order[i];
    if (previous < 0 || previous >= n || current < 0 || current >= n) throw new RangeError('ordered index outside record storage');
    if (wordColumns.some((column) => BigInt(column[previous]) !== BigInt(column[current]))) group += 1n;
    ids.push(group);
  }
  return ids;
}
