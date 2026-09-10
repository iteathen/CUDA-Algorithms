import { compileDeviceProgram } from 'cuda-js';
import { deviceProgramRequest } from './device-program.mjs';

const U32_BYTES = 4;
const DEFAULT_BLOCK_SIZE = 128;

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer in this experiment`);
  }
  return value;
}

function byteLengthU32(count, label) {
  const bytes = count * U32_BYTES;
  if (!Number.isSafeInteger(bytes)) throw new RangeError(`${label} byte length exceeds safe integer range`);
  return bytes;
}

function binding(name) {
  return Object.freeze({ binding: name });
}

function kernelByName(compiled, name) {
  const kernel = compiled.deviceProgram.kernels.find((entry) => entry.name === name);
  if (!kernel) throw new Error(`compiled Device-JS program is missing kernel ${name}`);
  return kernel;
}

export async function prepareStableSelectU32(runtime, options) {
  const inputCapacity = positiveSafeInteger(options?.inputCapacity, 'inputCapacity');
  const outputCapacity = positiveSafeInteger(options?.outputCapacity, 'outputCapacity');
  const blockSize = positiveSafeInteger(options?.blockSize ?? DEFAULT_BLOCK_SIZE, 'blockSize');
  if (blockSize > 1024) throw new RangeError('blockSize exceeds CUDA architectural thread-block ceiling');

  const inputBytes = byteLengthU32(inputCapacity, 'input');
  const outputBytes = byteLengthU32(outputCapacity, 'output');
  const gridX = Math.ceil(inputCapacity / blockSize);

  const compiled = await compileDeviceProgram(runtime, deviceProgramRequest);
  const artifact = compiled.compiler.artifact;
  if (artifact.format !== 'ptx' && artifact.format !== 'cubin') {
    throw new Error(`unexpected executable artifact format: ${artifact.format}`);
  }

  const module = await runtime.loadModule({ format: artifact.format, bytes: artifact.bytes });
  const resetKernel = kernelByName(compiled, 'resetControl');
  const scanKernel = kernelByName(compiled, 'exclusiveFlagScanU32');
  const selectKernel = kernelByName(compiled, 'selectIndicesFromScanU32');

  const reset = await module.getFunction({ name: resetKernel.functionName, parameters: resetKernel.parameters });
  const scan = await module.getFunction({ name: scanKernel.functionName, parameters: scanKernel.parameters });
  const select = await module.getFunction({ name: selectKernel.functionName, parameters: selectKernel.parameters });

  const prepared = await runtime.prepareOperationDag({
    nodes: [
      {
        id: 'reset',
        function: reset,
        grid: { x: 1, y: 1, z: 1 },
        block: { x: 1, y: 1, z: 1 },
        arguments: [binding('status'), binding('outputCount')],
        accesses: [
          { argumentIndex: 0, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
          { argumentIndex: 1, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
        ],
      },
      {
        id: 'scan',
        after: ['reset'],
        function: scan,
        grid: { x: gridX, y: 1, z: 1 },
        block: { x: blockSize, y: 1, z: 1 },
        arguments: [
          binding('flags'),
          binding('prefix'),
          binding('activeCount'),
          inputCapacity,
          binding('status'),
        ],
        accesses: [
          { argumentIndex: 0, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
          { argumentIndex: 1, byteOffset: 0, byteLength: inputBytes, mode: 'write' },
          { argumentIndex: 2, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
          { argumentIndex: 4, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
        ],
      },
      {
        id: 'select',
        after: ['scan'],
        function: select,
        grid: { x: gridX, y: 1, z: 1 },
        block: { x: blockSize, y: 1, z: 1 },
        arguments: [
          binding('flags'),
          binding('prefix'),
          binding('outputIndices'),
          binding('outputCount'),
          binding('activeCount'),
          inputCapacity,
          outputCapacity,
          binding('status'),
        ],
        accesses: [
          { argumentIndex: 0, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
          { argumentIndex: 1, byteOffset: 0, byteLength: inputBytes, mode: 'read' },
          { argumentIndex: 2, byteOffset: 0, byteLength: outputBytes, mode: 'write' },
          { argumentIndex: 3, byteOffset: 0, byteLength: U32_BYTES, mode: 'write' },
          { argumentIndex: 4, byteOffset: 0, byteLength: U32_BYTES, mode: 'read' },
          { argumentIndex: 7, byteOffset: 0, byteLength: U32_BYTES, mode: 'read-write' },
        ],
      },
    ],
  });

  return Object.freeze({
    inputCapacity,
    outputCapacity,
    blockSize,
    compiled,
    module,
    functions: Object.freeze([reset, scan, select]),
    prepared,
  });
}

export async function closePreparedStableSelect(preparedSlice) {
  const failures = [];
  const closeOne = async (label, resource) => {
    try {
      await resource.close();
    } catch (error) {
      failures.push({ label, error });
    }
  };

  await closeOne('prepared', preparedSlice.prepared);
  for (let i = preparedSlice.functions.length - 1; i >= 0; i -= 1) {
    await closeOne(`function-${i}`, preparedSlice.functions[i]);
  }
  await closeOne('module', preparedSlice.module);

  if (failures.length > 0) {
    throw new AggregateError(failures.map((entry) => entry.error), `failed to close ${failures.map((entry) => entry.label).join(', ')}`);
  }
}
