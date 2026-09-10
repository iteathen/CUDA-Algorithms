import { inspectDeviceProgram } from 'cuda-js';
import { deviceProgramRequest } from './device-program.mjs';

const inspected = inspectDeviceProgram(deviceProgramRequest);
console.log(JSON.stringify({
  contract: inspected.deviceProgram.contract,
  sha256: inspected.deviceProgram.sha256,
  kernels: inspected.deviceProgram.kernels,
  publicHelperUsage: inspected.inspection.publicHelperUsage,
}, null, 2));
