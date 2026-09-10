import { compileDeviceProgram } from 'cuda-js';
import { deviceProgramRequest } from './device-program.mjs';

const U32_BYTES = 4;
const DEFAULT_BLOCK_SIZE = 128;

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer in this experiment`);
  return value;
}

function u32Bytes(count, label) {
  const bytes = count * U32_BYTES;
  if (!Number.isSafeInteger(bytes)) throw new RangeError(`${label} byte length exceeds safe integer range`);
  return bytes;
}

const binding = (name) => Object.freeze({ binding: name });

function kernelByName(compiled, name) {
  const kernel = compiled.deviceProgram.kernels.find((entry) => entry.name === name);
  if (!kernel) throw new Error(`compiled Device-JS program is missing kernel ${name}`);
  return kernel;
}

export async function prepareTwoWordStableOrderU32(runtime, options) {
  const recordCapacity = positiveSafeInteger(options?.recordCapacity, 'recordCapacity');
  const indexCapacity = positiveSafeInteger(options?.indexCapacity, 'indexCapacity');
  const blockSize = positiveSafeInteger(options?.blockSize ?? DEFAULT_BLOCK_SIZE, 'blockSize');
  if (blockSize > 1024) throw new RangeError('blockSize exceeds CUDA architectural thread-block ceiling');

  const recordBytes = u32Bytes(recordCapacity, 'record');
  const indexBytes = u32Bytes(indexCapacity, 'index');
  const gridX = Math.ceil(indexCapacity / blockSize);

  const compiled = await compileDeviceProgram(runtime, deviceProgramRequest);
  const artifact = compiled.compiler.artifact;
  if (artifact.format !== 'ptx' && artifact.format !== 'cubin') throw new Error(`unexpected executable artifact format: ${artifact.format}`);

  const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes });
  const resetKernel = kernelByName(compiled, 'resetOrderingStatus');
  const orderKernel = kernelByName(compiled, 'stableOrderIndicesByKeyU32');
  const reset = await module.getFunction({ name: resetKernel.functionName, parameters: resetKernel.parameters });
  const order = await module.getFunction({ name: orderKernel.functionName, parameters: orderKernel.parameters });

  const orderAccesses = (keyArgumentIndex, inputArgumentIndex, outputArgumentIndex, activeArgumentIndex, statusArgumentIndex) => [
    { argumentIndex: keyArgumentIndex, byteOffset: 0, byteLength: recordBytes, mode: 'read' },
    { argumentIndex: inputArgumentIndex, byteOffset: 0, byteLength: indexBytes, mode: 'read' },
    { argumentIndex: outputArgumentIndex, byteOffset: 0, byteLength: indexBytes, mode: 'write' },
    { argumentIndex: activeArgumentIndex, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
    { argumentIndex: statusArgumentIndex, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
  ];

  const commonGrid = { x: gridX, y: 1, z: 1 };
  const commonBlock = { x: blockSize, y: 1, z: 1 };

  const prepared = await runtime.prepareOperationDag({
    nodes: [
      {
        id: 'reset',
        function: reset,
        grid: { x: 1, y: 1, z: 1 },
        block: { x: 1, y: 1, z: 1 },
        arguments: [binding('status')],
        accesses: [{ argumentIndex: 0, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' }],
      },
      {
        id: 'order-low-word',
        after: ['reset'],
        function: order,
        grid: commonGrid,
        block: commonBlock,
        arguments: [
          binding('lowKeys'),
          binding('indicesA'),
          binding('indicesB'),
          binding('activeCount'),
          recordCapacity,
          indexCapacity,
          indexCapacity,
          binding('status'),
        ],
        accesses: orderAccesses(0, 1, 2, 3, 7),
      },
      {
        id: 'order-high-word',
        after: ['order-low-word'],
        function: order,
        grid: commonGrid,
        block: commonBlock,
        arguments: [
          binding('highKeys'),
          binding('indicesB'),
          binding('indicesA'),
          binding('activeCount'),
          recordCapacity,
          indexCapacity,
          indexCapacity,
          binding('status'),
        ],
        accesses: orderAccesses(0, 1, 2, 3, 7),
      },
    ],
  });

  return Object.freeze({ recordCapacity, indexCapacity, blockSize, compiled, module, functions: Object.freeze([reset, order]), prepared });
}

export async function closePreparedTwoWordOrder(preparedSlice) {
  const errors = [];
  const closeOne = async (resource) => {
    try { await resource.close(); } catch (error) { errors.push(error); }
  };
  await closeOne(preparedSlice.prepared);
  for (let i = preparedSlice.functions.length - 1; i >= 0; i -= 1) await closeOne(preparedSlice.functions[i]);
  await closeOne(preparedSlice.module);
  if (errors.length > 0) throw new AggregateError(errors, 'failed to close prepared ordering slice');
}
