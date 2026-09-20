# QCOM A630 MockGPU

Run tinygrad's QCOM runtime and compiled A630 instructions on a CPU, without a Qualcomm device. The mock implements KGSL requests, compute command packets, instruction execution, workgroup barriers, and float32/float16 2D images. It uses the existing Mesa instruction decoder.

Install the test and Mesa dependencies and a C compiler:

```sh
python -m pip install -e '.[testing_minimal,mesa]'
DEV=MOCK+QCOM:IR3 python -c 'from tinygrad import Tensor; print((Tensor([1., 2.]) + 1).tolist())'
```

QCOM MockGPU uses the compiled CPU host runtime because command submission calls `ioctl`. The emulated device and its mapped buffers share CPU memory. Other MockGPU backends retain their existing host-runtime defaults.

Run the focused tests:

```sh
DEV=MOCK+QCOM:IR3 python -m pytest -n12 test/mockgpu/qcom
DEV=MOCK+QCOM:IR3 IMAGE=1 python -m pytest -n12 test/mockgpu/qcom/test_qcom.py -k image
```

Image support must be enabled before device initialization. `IMAGE=1` adds the image alignment requirement to the QCOM target; setting it later does not configure the same target.

The OpenCL path uses the existing QCOM compiler support, including its qemu or Docker compiler server on non-aarch64 hosts:

```sh
DEV=MOCK+QCOM:CL python -m pytest -n12 test/mockgpu/qcom
DEV=MOCK+QCOM:CL IMAGE=1 FLOAT16=1 python -m pytest -n12 test/mockgpu/qcom/test_qcom.py -k image
```

Run the backend operation tests with the same settings as the MockGPU CI job:

```sh
DEV=MOCK+QCOM:IR3 TRANSCENDENTAL=2 FORWARD_ONLY=1 SKIP_SLOW_TEST=1 TEST_TIMEOUT=600 python -m pytest -n4 \
  test/backend/test_ops.py::TestOps::test_add test/backend/test_ops.py::TestOps::test_mul \
  test/backend/test_ops.py::TestOps::test_where test/backend/test_ops.py::TestOps::test_cast \
  test/backend/test_ops.py::TestOps::test_sum test/backend/test_ops.py::TestOps::test_max \
  test/backend/test_ops.py::TestOps::test_gemm test/backend/test_ops.py::TestOps::test_conv2d
DEV=MOCK+QCOM:IR3 TRANSCENDENTAL=2 FORWARD_ONLY=1 SKIP_SLOW_TEST=1 TEST_TIMEOUT=240 python -m pytest -n1 \
  test/backend/test_ops.py::TestOps::test_sin test/backend/test_ops.py::TestOps::test_cos \
  test/backend/test_ops.py::TestOps::test_tan test/backend/test_ops.py::TestOps::test_atan
DEV=MOCK+QCOM:CL FORWARD_ONLY=1 SKIP_SLOW_TEST=1 TEST_TIMEOUT=600 python -m pytest -n4 \
  test/backend/test_ops.py::TestOps::test_add test/backend/test_ops.py::TestOps::test_mul \
  test/backend/test_ops.py::TestOps::test_where test/backend/test_ops.py::TestOps::test_cast \
  test/backend/test_ops.py::TestOps::test_sum test/backend/test_ops.py::TestOps::test_max \
  test/backend/test_ops.py::TestOps::test_gemm test/backend/test_ops.py::TestOps::test_conv2d
DEV=MOCK+QCOM:CL FORWARD_ONLY=1 SKIP_SLOW_TEST=1 TEST_TIMEOUT=240 python -m pytest -n1 \
  test/backend/test_ops.py::TestOps::test_sin test/backend/test_ops.py::TestOps::test_cos \
  test/backend/test_ops.py::TestOps::test_tan test/backend/test_ops.py::TestOps::test_atan
```

For a full local soak, run `test/backend/test_ops.py` at `-n4` with `TEST_TIMEOUT=600`. On the
scalar IR3 interpreter this currently takes about 41 minutes (356 passed, 68 skipped on the
development machine), so CI uses the focused emulator suite plus the representative operations above.
`TestOps.test_all_large` is also soak coverage: it passes in about nine minutes locally, while the
focused suite's compiled 128-lane shared-memory kernel covers the same multi-wave barrier behavior with a smaller input.

The IR3 invocation uses software transcendental lowering because native NIR large-angle trigonometry has limited precision. These commands exercise forward calculations and retain the existing slow-test skips. The interpreter models functional results rather than instruction timing, caches, or graphics operations. Unsupported instruction/addressing modes raise errors. GPU execution failures are retained across the C callback and reported on synchronization, with the original exception in the cause chain.

The scalar executor supports lane-local instructions plus workgroup shared memory and barrier rendezvous. It accepts workgroup-uniform,
read-only, full-width shared-register launch metadata. It rejects half-width shared-register operands because their packed aliasing is not modeled.
Dynamic shared-register writes and subgroup-dependent forms are also rejected instead of being treated as lane-local; supporting them requires
wave-aware active-mask and reconvergence semantics.
