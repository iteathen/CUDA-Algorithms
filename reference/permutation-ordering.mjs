function widthMax(width) {
  if (width !== 32 && width !== 64) throw new RangeError(`unsupported width: ${width}`);
  return (1n << BigInt(width)) - 1n;
}

function asUnsigned(value, width, label) {
  const v = typeof value === 'bigint' ? value : BigInt(value);
  if (v < 0n || v > widthMax(width)) throw new RangeError(`${label} is outside u${width}`);
  return v;
}

function keyIndex(value, length, width, label) {
  const v = asUnsigned(value, width, label);
  if (v >= BigInt(length)) throw new RangeError(`${label} is outside key storage`);
  return Number(v);
}

export function stableOrderIndicesByKey(keys, indices = null, options = {}) {
  const { keyWidth = 32, indexWidth = 32 } = options;
  widthMax(keyWidth);
  widthMax(indexWidth);

  const order = indices === null
    ? Array.from({ length: keys.length }, (_, i) => asUnsigned(BigInt(i), indexWidth, `index ${i}`))
    : indices.map((value, i) => asUnsigned(value, indexWidth, `indices[${i}]`));

  const rows = order.map((index, ordinal) => ({
    index,
    ordinal,
    key: asUnsigned(keys[keyIndex(index, keys.length, indexWidth, `indices[${ordinal}]`)], keyWidth, 'key'),
  }));

  rows.sort((a, b) => {
    if (a.key < b.key) return -1;
    if (a.key > b.key) return 1;
    return a.ordinal - b.ordinal;
  });

  return rows.map((row) => row.index);
}

export function stableLexicographicOrderIndices(wordColumns, indices = null, options = {}) {
  if (!Array.isArray(wordColumns) || wordColumns.length === 0) throw new TypeError('wordColumns must be non-empty');
  const n = wordColumns[0].length;
  if (wordColumns.some((column) => column.length !== n)) throw new RangeError('word columns must have equal length');

  let order = indices === null
    ? Array.from({ length: n }, (_, i) => BigInt(i))
    : [...indices];

  for (let column = wordColumns.length - 1; column >= 0; column -= 1) {
    order = stableOrderIndicesByKey(wordColumns[column], order, options);
  }

  return order;
}
