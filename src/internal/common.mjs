export const U32_BYTES = 4;
export const DEFAULT_BLOCK_SIZE = 128;
export const CUDA_THREAD_BLOCK_CEILING = 1024;
export const PREPARED_KERNEL_NODE_CEILING = 32;

export function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

export function blockSize(value = DEFAULT_BLOCK_SIZE) {
  const normalized = positiveSafeInteger(value, 'blockSize');
  if (normalized > CUDA_THREAD_BLOCK_CEILING) throw new RangeError('blockSize exceeds CUDA architectural thread-block ceiling');
  return normalized;
}

export function u32Bytes(count, label) {
  const bytes = count * U32_BYTES;
  if (!Number.isSafeInteger(bytes)) throw new RangeError(`${label} byte length exceeds safe integer range`);
  return bytes;
}

export function binding(name) {
  return Object.freeze({ binding: name });
}

export function kernelByName(compiled, name) {
  const kernel = compiled.deviceProgram.kernels.find((entry) => entry.name === name);
  if (!kernel) throw new Error(`compiled Device-JS program is missing kernel ${name}`);
  return kernel;
}

function accessAllows(actual, required) {
  if (actual === 'read-write') return true;
  return actual === required;
}

export function requireU32View(view, minimumElements, label, requiredAccess) {
  if (!view || view.kind !== 'device-view') throw new TypeError(`${label} must be a CUDA-JS device view`);
  if (view.dtype !== 'u32') throw new TypeError(`${label} must have dtype u32`);
  if (!Number.isSafeInteger(view.elementCount) || view.elementCount < minimumElements) {
    throw new RangeError(`${label} does not cover the required element range`);
  }
  if (!accessAllows(view.access, requiredAccess)) throw new TypeError(`${label} does not provide required ${requiredAccess} access`);
  return view;
}

export async function closeResources(resources, label = 'CUDA-Algorithms plan') {
  const failures = [];
  for (let i = resources.length - 1; i >= 0; i -= 1) {
    try {
      await resources[i].close();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length > 0) throw new AggregateError(failures, `${label} cleanup failed`);
}
