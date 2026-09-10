const WIDTHS = new Set([32, 64]);
const OPS = new Set(['add', 'min', 'max', 'bitAnd', 'bitOr', 'bitXor']);

function assertWidth(width) {
  if (!WIDTHS.has(width)) throw new RangeError(`unsupported width: ${width}`);
  return width;
}

function modulus(width) {
  return 1n << BigInt(assertWidth(width));
}

function maxValue(width) {
  return modulus(width) - 1n;
}

function asUnsigned(value, width, label = 'value') {
  const v = typeof value === 'bigint' ? value : BigInt(value);
  const max = maxValue(width);
  if (v < 0n || v > max) throw new RangeError(`${label} is outside u${width}`);
  return v;
}

function asCount(value, label = 'count') {
  const v = typeof value === 'bigint' ? value : BigInt(value);
  if (v < 0n) throw new RangeError(`${label} must be non-negative`);
  return v;
}

function toArrayIndex(value, length, label = 'index') {
  const v = asCount(value, label);
  if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new RangeError(`${label} exceeds JS reference index range`);
  const n = Number(v);
  if (n >= length) throw new RangeError(`${label} ${n} is outside input length ${length}`);
  return n;
}

function activeLength(active, capacity, available) {
  const cap = asCount(capacity ?? available, 'capacity');
  const act = asCount(active ?? available, 'active');
  if (cap > BigInt(Number.MAX_SAFE_INTEGER) || act > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError('reference arrays require safe-integer active/capacity');
  }
  if (act > cap) throw new RangeError('active exceeds capacity');
  if (cap > BigInt(available)) throw new RangeError('capacity exceeds available reference storage');
  return Number(act);
}

function combine(a, b, width, op) {
  if (!OPS.has(op)) throw new RangeError(`unsupported op: ${op}`);
  switch (op) {
    case 'add':
      return (a + b) % modulus(width);
    case 'min':
      return a < b ? a : b;
    case 'max':
      return a > b ? a : b;
    case 'bitAnd':
      return a & b;
    case 'bitOr':
      return a | b;
    case 'bitXor':
      return a ^ b;
  }
}

export function scanUnsigned(input, options = {}) {
  const {
    width = 32,
    mode = 'exclusive',
    op = 'add',
    init = 0n,
    active,
    capacity,
  } = options;
  assertWidth(width);
  if (mode !== 'inclusive' && mode !== 'exclusive') throw new RangeError(`unsupported scan mode: ${mode}`);
  const n = activeLength(active, capacity, input.length);
  const values = input.slice(0, n).map((v, i) => asUnsigned(v, width, `input[${i}]`));
  if (n === 0) return [];

  const out = new Array(n);
  if (mode === 'inclusive') {
    let acc = values[0];
    out[0] = acc;
    for (let i = 1; i < n; i += 1) {
      acc = combine(acc, values[i], width, op);
      out[i] = acc;
    }
    return out;
  }

  let acc = asUnsigned(init, width, 'init');
  for (let i = 0; i < n; i += 1) {
    out[i] = acc;
    acc = combine(acc, values[i], width, op);
  }
  return out;
}

export function reduceUnsigned(input, options = {}) {
  const {
    width = 32,
    op = 'add',
    init = 0n,
    active,
    capacity,
  } = options;
  assertWidth(width);
  const n = activeLength(active, capacity, input.length);
  let acc = asUnsigned(init, width, 'init');
  for (let i = 0; i < n; i += 1) {
    acc = combine(acc, asUnsigned(input[i], width, `input[${i}]`), width, op);
  }
  return acc;
}

export function selectIndices(flags, options = {}) {
  const {
    indexWidth = 32,
    active,
    capacity,
    outputCapacity = null,
  } = options;
  assertWidth(indexWidth);
  const n = activeLength(active, capacity, flags.length);
  const indices = [];
  for (let i = 0; i < n; i += 1) {
    const flag = typeof flags[i] === 'bigint' ? flags[i] : BigInt(flags[i]);
    if (flag !== 0n && flag !== 1n) throw new RangeError(`flags[${i}] must be 0 or 1`);
    if (flag === 1n) indices.push(asUnsigned(BigInt(i), indexWidth, `index ${i}`));
  }
  const outputCount = BigInt(indices.length);
  if (outputCapacity !== null && outputCount > asCount(outputCapacity, 'outputCapacity')) {
    return { status: 'capacity-exhausted', indices: [], outputCount };
  }
  return { status: 'ok', indices, outputCount };
}

export function gather(input, indices, options = {}) {
  const { activeIndices, capacity = indices.length } = options;
  const n = activeLength(activeIndices, capacity, indices.length);
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = input[toArrayIndex(indices[i], input.length, `indices[${i}]`)];
  }
  return out;
}

function selectedBits(key, width, beginBit, endBit) {
  const w = assertWidth(width);
  if (!Number.isInteger(beginBit) || !Number.isInteger(endBit) || beginBit < 0 || endBit < beginBit || endBit > w) {
    throw new RangeError(`invalid bit range [${beginBit}, ${endBit}) for u${w}`);
  }
  if (beginBit === endBit) return 0n;
  const bits = BigInt(endBit - beginBit);
  const mask = (1n << bits) - 1n;
  return (key >> BigInt(beginBit)) & mask;
}

export function stableRadixSortPairs(keys, indices = null, options = {}) {
  const {
    width = 32,
    direction = 'asc',
    beginBit = 0,
    endBit = width,
    active,
    capacity,
    indexWidth = 32,
  } = options;
  assertWidth(width);
  assertWidth(indexWidth);
  if (direction !== 'asc' && direction !== 'desc') throw new RangeError(`unsupported direction: ${direction}`);
  const n = activeLength(active, capacity, keys.length);
  if (indices !== null && indices.length < n) throw new RangeError('indices shorter than active keys');

  const decorated = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const key = asUnsigned(keys[i], width, `keys[${i}]`);
    const index = indices === null ? asUnsigned(BigInt(i), indexWidth, `index ${i}`) : asUnsigned(indices[i], indexWidth, `indices[${i}]`);
    decorated[i] = { key, index, selected: selectedBits(key, width, beginBit, endBit), ordinal: i };
  }

  decorated.sort((a, b) => {
    if (a.selected < b.selected) return direction === 'asc' ? -1 : 1;
    if (a.selected > b.selected) return direction === 'asc' ? 1 : -1;
    return a.ordinal - b.ordinal;
  });

  return {
    keys: decorated.map((x) => x.key),
    indices: decorated.map((x) => x.index),
  };
}

export function runLengthEncode(keys, options = {}) {
  const {
    width = 32,
    active,
    capacity,
    runLengthWidth = 64,
    outputCapacity = null,
  } = options;
  assertWidth(width);
  assertWidth(runLengthWidth);
  const n = activeLength(active, capacity, keys.length);
  if (n === 0) return { status: 'ok', uniqueKeys: [], runLengths: [], outputRunCount: 0n };

  const uniqueKeys = [];
  const runLengths = [];
  let current = asUnsigned(keys[0], width, 'keys[0]');
  let length = 1n;
  for (let i = 1; i < n; i += 1) {
    const key = asUnsigned(keys[i], width, `keys[${i}]`);
    if (key === current) {
      length += 1n;
    } else {
      uniqueKeys.push(current);
      runLengths.push(asUnsigned(length, runLengthWidth, 'run length'));
      current = key;
      length = 1n;
    }
  }
  uniqueKeys.push(current);
  runLengths.push(asUnsigned(length, runLengthWidth, 'run length'));

  const outputRunCount = BigInt(uniqueKeys.length);
  if (outputCapacity !== null && outputRunCount > asCount(outputCapacity, 'outputCapacity')) {
    return { status: 'capacity-exhausted', uniqueKeys: [], runLengths: [], outputRunCount };
  }
  return { status: 'ok', uniqueKeys, runLengths, outputRunCount };
}

export function reduceByKey(keys, values, options = {}) {
  const {
    keyWidth = 32,
    valueWidth = 32,
    op = 'add',
    initPolicy = 'first',
    init = 0n,
    active,
    capacity,
    outputCapacity = null,
  } = options;
  assertWidth(keyWidth);
  assertWidth(valueWidth);
  const n = activeLength(active, capacity, Math.min(keys.length, values.length));
  if (n === 0) return { status: 'ok', uniqueKeys: [], aggregates: [], outputRunCount: 0n };
  if (initPolicy !== 'first' && initPolicy !== 'identity') throw new RangeError(`unsupported initPolicy: ${initPolicy}`);

  const uniqueKeys = [];
  const aggregates = [];
  let currentKey = asUnsigned(keys[0], keyWidth, 'keys[0]');
  let acc = initPolicy === 'first'
    ? asUnsigned(values[0], valueWidth, 'values[0]')
    : combine(asUnsigned(init, valueWidth, 'init'), asUnsigned(values[0], valueWidth, 'values[0]'), valueWidth, op);

  for (let i = 1; i < n; i += 1) {
    const key = asUnsigned(keys[i], keyWidth, `keys[${i}]`);
    const value = asUnsigned(values[i], valueWidth, `values[${i}]`);
    if (key === currentKey) {
      acc = combine(acc, value, valueWidth, op);
    } else {
      uniqueKeys.push(currentKey);
      aggregates.push(acc);
      currentKey = key;
      acc = initPolicy === 'first' ? value : combine(asUnsigned(init, valueWidth, 'init'), value, valueWidth, op);
    }
  }
  uniqueKeys.push(currentKey);
  aggregates.push(acc);

  const outputRunCount = BigInt(uniqueKeys.length);
  if (outputCapacity !== null && outputRunCount > asCount(outputCapacity, 'outputCapacity')) {
    return { status: 'capacity-exhausted', uniqueKeys: [], aggregates: [], outputRunCount };
  }
  return { status: 'ok', uniqueKeys, aggregates, outputRunCount };
}

export function stableLexicographicIndexOrder(wordColumns, options = {}) {
  const { wordWidth = 32, indexWidth = 32 } = options;
  assertWidth(wordWidth);
  assertWidth(indexWidth);
  if (!Array.isArray(wordColumns) || wordColumns.length === 0) throw new TypeError('wordColumns must be a non-empty array');
  const n = wordColumns[0].length;
  for (const column of wordColumns) {
    if (column.length !== n) throw new RangeError('word columns must have equal lengths');
  }

  let order = Array.from({ length: n }, (_, i) => asUnsigned(BigInt(i), indexWidth, `index ${i}`));
  for (let column = wordColumns.length - 1; column >= 0; column -= 1) {
    const passKeys = order.map((idx) => wordColumns[column][toArrayIndex(idx, n, 'sort index')]);
    order = stableRadixSortPairs(passKeys, order, { width: wordWidth, indexWidth }).indices;
  }
  return order;
}

export function independentLexicographicIndexOrder(wordColumns, options = {}) {
  const { wordWidth = 32, indexWidth = 32 } = options;
  assertWidth(wordWidth);
  assertWidth(indexWidth);
  if (!Array.isArray(wordColumns) || wordColumns.length === 0) throw new TypeError('wordColumns must be a non-empty array');
  const n = wordColumns[0].length;
  const rows = Array.from({ length: n }, (_, i) => ({
    index: i,
    words: wordColumns.map((column, c) => asUnsigned(column[i], wordWidth, `wordColumns[${c}][${i}]`)),
  }));
  rows.sort((a, b) => {
    for (let c = 0; c < a.words.length; c += 1) {
      if (a.words[c] < b.words[c]) return -1;
      if (a.words[c] > b.words[c]) return 1;
    }
    return a.index - b.index;
  });
  return rows.map((row) => asUnsigned(BigInt(row.index), indexWidth, `index ${row.index}`));
}

export function validateDeviceExtent(count, capacity, width = 32) {
  assertWidth(width);
  const c = asUnsigned(count, width, 'count');
  const cap = asCount(capacity, 'capacity');
  return c <= cap
    ? { status: 'ok', active: c }
    : { status: 'invalid-extent', active: null };
}
