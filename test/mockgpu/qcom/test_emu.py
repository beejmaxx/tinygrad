import base64, ctypes, os, struct, unittest
from unittest.mock import patch
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, cast
from tinygrad.device import TinyELF
from tinygrad.codegen import to_program
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.nir import IR3Renderer, _nload_img, nalu, nchannel, nimm, nir_instr, nlid, nsrc, nstore, nstore_img
from tinygrad.runtime.support.compiler_mesa import IR3Compiler
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import QCOMDevice, QCOMProgramData, pkt4_hdr, pkt7_hdr
from tinygrad.uop.ops import KernelInfo, UOp
from test.mockgpu.qcom.emu import Image, Memory, decode, f32bits, run_scalar, run_thread, run_workgroup
from test.mockgpu.qcom.qcomgpu import QCOMGPU

_nread_first = nir_instr(nc=1, bs=lambda src: src.bit_size, num_components=1, srcs=lambda src:[nsrc(src)])(
  lambda b,src: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_read_first_invocation))

class TestA630CommandPackets(unittest.TestCase):
  @staticmethod
  def image_fixture(texture:bool=False):
    storage = (ctypes.c_ubyte*(4*64+31))()
    pointer = (ctypes.addressof(storage)+31) & ~31
    word0 = mesa.FMT6_32_32_32_32_FLOAT << mesa.A6XX_TEX_CONST_0_FMT__SHIFT
    if texture: word0 |= (1 << 7) | (2 << 10) | (3 << 13) | 8
    descriptor = (word0, 4 | 4 << 15, mesa.A6XX_TEX_2D << mesa.A6XX_TEX_CONST_2_TYPE__SHIFT | 64 << 7,
                  0, pointer & 0xffffffff, pointer >> 32, 0x40000000, 13, 0, 0, 0, 0, 0, 0, 0, 0)
    return storage,pointer,descriptor

  @staticmethod
  def submit_words(*words:int):
    command = (ctypes.c_uint32*len(words))(*words)
    pointer = ctypes.addressof(command)
    QCOMGPU(lambda: ((pointer, ctypes.sizeof(command)),)).submit(pointer, ctypes.sizeof(command))

  def test_packet_parity_is_checked(self):
    with self.assertRaisesRegex(ValueError, 'type-7 packet header'):
      self.submit_words(pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) ^ (1 << 23))
    with self.assertRaisesRegex(ValueError, 'type-4 packet header'):
      self.submit_words(pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 1) ^ (1 << 27), 0)

  def test_packet_reserved_bits_are_checked(self):
    for bit in (14, 24, 27):
      with self.subTest(bit=bit), self.assertRaisesRegex(ValueError, 'type-7 packet header'):
        self.submit_words(pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) | (1 << bit))

  def test_late_packet_error_does_not_publish_prefix(self):
    signal = ctypes.c_uint32(0x11111111)
    pointer = ctypes.addressof(signal)
    words = [pkt7_hdr(mesa.CP_EVENT_WRITE, 4), mesa.CACHE_FLUSH_TS, pointer & 0xffffffff, pointer >> 32, 0x22222222,
             pkt7_hdr(0x7f, 0)]
    command = (ctypes.c_uint32*len(words))(*words)
    address = ctypes.addressof(command)
    gpu = QCOMGPU(lambda: ((address, ctypes.sizeof(command)), (pointer, 4)))
    with self.assertRaisesRegex(ValueError, 'Unsupported A630 command'): gpu.submit(address, ctypes.sizeof(command))
    self.assertEqual(signal.value, 0x11111111)

  def test_truncated_packet_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'Truncated A630 command packet'):
      self.submit_words(pkt7_hdr(mesa.CP_SET_MARKER, 1))

  def test_empty_command_buffer_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'invalid size'): self.submit_words()

  def test_unsupported_control_modes_are_rejected(self):
    gpu, memory = QCOMGPU(lambda: ()), Memory(())
    with self.assertRaisesRegex(ValueError, 'register-to-memory'):
      gpu.packet(mesa.CP_REG_TO_MEM, [0, 0, 0], memory)
    with self.assertRaisesRegex(ValueError, 'execute control'):
      gpu.packet(mesa.CP_EXEC_CS, [1, 1, 1, 1], memory)

  def test_wait_function_is_honored(self):
    value = ctypes.c_uint32(7)
    pointer = ctypes.addressof(value)
    memory = Memory(((pointer, ctypes.sizeof(value)),))
    gpu = QCOMGPU(lambda: memory.ranges)
    def wait(function:int, reference:int):
      control = function << mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT | mesa.POLL_MEMORY << mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
      gpu.packet(mesa.CP_WAIT_REG_MEM, [control, pointer & 0xffffffff, pointer >> 32, reference, 0xffffffff, 0], memory)
    wait(mesa.WRITE_EQ, 7)
    wait(mesa.WRITE_NE, 8)
    with self.assertRaisesRegex(RuntimeError, 'unsatisfied memory dependency'): wait(mesa.WRITE_EQ, 8)
    signed_control = (mesa.WRITE_GE << mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT |
                      mesa.POLL_MEMORY << mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT | mesa.CP_WAIT_REG_MEM_0_SIGNED_COMPARE)
    with self.assertRaisesRegex(ValueError, 'memory wait mode'):
      gpu.packet(mesa.CP_WAIT_REG_MEM, [signed_control, pointer & 0xffffffff, pointer >> 32, 7, 0xffffffff, 0], memory)

  def test_tiled_images_fail_closed(self):
    _storage,pointer,descriptor = self.image_fixture()
    gpu = QCOMGPU(lambda: ((pointer, 4*64),))
    gpu.registers[mesa.REG_A6XX_SP_CS_CONFIG] = 1 << mesa.A6XX_SP_CS_CONFIG_NUAV__SHIFT
    gpu.state_blocks[mesa.SB6_CS_SHADER,mesa.ST6_UAV,0] = (descriptor[0] | mesa.TILE6_3, *descriptor[1:])
    with self.assertRaisesRegex(ValueError, 'image layout'): gpu.images(Memory(((pointer, 4*64),)), False)

  def test_unsupported_image_sampler_fails_closed(self):
    _storage,pointer,descriptor = self.image_fixture(texture=True)
    gpu = QCOMGPU(lambda: ((pointer, 4*64),))
    gpu.registers[mesa.REG_A6XX_SP_CS_CONFIG] = (1 << mesa.A6XX_SP_CS_CONFIG_NTEX__SHIFT | 1 << mesa.A6XX_SP_CS_CONFIG_NSAMP__SHIFT)
    gpu.state_blocks[mesa.SB6_CS_TEX,mesa.ST_CONSTANTS,0] = descriptor
    gpu.state_blocks[mesa.SB6_CS_TEX,mesa.ST_SHADER,0] = (1, 0x30, 0, 0)
    with self.assertRaisesRegex(ValueError, 'image sampler'): gpu.samplers(Memory(((pointer, 4*64),)), 1)

  def test_nonzero_image_border_fails_closed(self):
    _storage,pointer,descriptor = self.image_fixture(texture=True)
    _border_storage = (ctypes.c_ubyte*(4096+63))()
    border = (ctypes.addressof(_border_storage)+63) & ~63
    ctypes.memset(border, 1, 1)
    gpu = QCOMGPU(lambda: ((pointer, 4*64), (border, 4096)))
    gpu.registers[mesa.REG_A6XX_SP_CS_CONFIG] = (1 << mesa.A6XX_SP_CS_CONFIG_NTEX__SHIFT | 1 << mesa.A6XX_SP_CS_CONFIG_NSAMP__SHIFT)
    gpu.registers[mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE] = border & 0xffffffff
    gpu.registers[mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE+1] = border >> 32
    gpu.state_blocks[mesa.SB6_CS_TEX,mesa.ST_CONSTANTS,0] = descriptor
    gpu.state_blocks[mesa.SB6_CS_TEX,mesa.ST_SHADER,0] = (3 << 5 | 3 << 8 | 3 << 11, 0x30, 0, 0)
    with self.assertRaisesRegex(ValueError, 'border color'): gpu.samplers(Memory(((pointer, 4*64), (border, 4096))), 1)

  @unittest.skipUnless('mesa' in mesa.dll._loaded_, 'requires the optional Mesa package')
  def test_instruction_cannot_select_reserved_sampler(self):
    program = struct.pack('<Q', 0xa0001f0000800001)  # isam with direct texture 0 and sampler 4.
    with self.assertRaisesRegex(ValueError, 'unavailable sampler'):
      QCOMGPU.validate_image_accesses(program, 1, 0, frozenset({0}))

  @unittest.skipUnless('mesa' in mesa.dll._loaded_, 'requires the optional Mesa package')
  def test_indirect_texture_selection_fails_closed(self):
    program = struct.pack('<Q', 0xa0081f0000800001)  # isam.s2en selects texture/sampler through registers.
    with self.assertRaisesRegex(ValueError, 'addressing mode'):
      QCOMGPU.validate_image_accesses(program, 1, 0, frozenset({0}))

  def test_launch_register_destinations_cannot_overlap(self):
    def config(wgid:int=0xfc, wgsz:int=0xfc, wgoff:int=0xfc, lid:int=0xfc):
      return wgid | wgsz << 8 | wgoff << 16 | lid << 24
    gpu = QCOMGPU(lambda: ())
    gpu.registers[mesa.REG_A6XX_SP_CS_WGE_CNTL] = 0xfc
    gpu.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0] = config(wgid=48*4, wgsz=48*4)
    with self.assertRaisesRegex(ValueError, 'launch metadata register .* overlaps'):
      gpu.execute((1, 1, 1), Memory(()))
    gpu.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0] = config(lid=48*4)
    with self.assertRaisesRegex(ValueError, 'local ID cannot target shared registers'):
      gpu.execute((1, 1, 1), Memory(()))
    gpu.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0] = config(wgid=56*4)
    with self.assertRaisesRegex(ValueError, 'outside the GPR files'):
      gpu.execute((1, 1, 1), Memory(()))

  @unittest.skipUnless('mesa' in mesa.dll._loaded_, 'requires the optional Mesa package')
  def test_uniform_launch_metadata_crosses_shared_boundary(self):
    def config(wgsz:int): return 0xfc | wgsz << 8 | 0xfc << 16 | 0xfc << 24
    program = ctypes.c_uint64(0x0300000000000000)
    pointer = ctypes.addressof(program)
    gpu = QCOMGPU(lambda: ((pointer, 8),))
    gpu.program_address, gpu.program_size = pointer, 8
    gpu.registers[mesa.REG_A6XX_SP_CS_NDRANGE_0] = (1 << mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT |
                                                    2 << mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT)
    gpu.registers[mesa.REG_A6XX_SP_CS_WGE_CNTL] = 0xfc << mesa.A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID__SHIFT
    cases = ((190, {190:1, 191:2}, {192:3}), (191, {191:1}, {192:2, 193:3}))
    for base,ordinary,shared in cases:
      with self.subTest(base=base), patch('test.mockgpu.qcom.qcomgpu.run_workgroup') as run:
        gpu.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0] = config(base)
        gpu.execute((1, 1, 1), Memory(((pointer, 8),)))
        self.assertEqual(len(run.call_args.args[3]), 6)
        self.assertTrue(all(registers == ordinary for registers in run.call_args.args[3]))
        self.assertEqual(run.call_args.kwargs['shared_registers'], shared)

@unittest.skipUnless('mesa' in mesa.dll._loaded_, 'requires the optional Mesa package')
class TestA630Workgroup(unittest.TestCase):
  END, BAR = 0x0300000000000000, 7 << 61
  @staticmethod
  def _compile_nir(renderer:IR3Renderer, label:bytes):
    builder = renderer.b
    mesa.nir_validate_shader(builder.shader, label)
    blob = mesa.struct_blob()
    mesa.nir_serialize(blob, builder.shader, False)
    try:
      source = base64.b64encode(ctypes.string_at(blob.data, blob.size)).decode()
      variant, state, immediates, program = IR3Compiler.unpack_lib(renderer.compiler.compile(source))
    finally:
      mesa.ralloc_free(builder.shader)
      ctypes.CDLL(None).free(blob.data)
    offset = state.allocs.max_const_offset_vec4 * 16
    constants = bytes(offset)+immediates
    allocation = state.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]
    wgsize = allocation.offset_vec4 * 4 + 8 if allocation.size_vec4 else 0xfc
    if wgsize != 0xfc: constants = constants.ljust((wgsize+3)*4, b'\0')
    return variant, struct.unpack('<'+'I'*(len(constants)//4), constants), program

  def test_uniform_barrier(self):
    run_workgroup(struct.pack('<2Q', self.BAR, self.END), (), Memory(()), [{}, {}], 0, 0)

  def test_unmatched_legacy_synchronization_rejects_reserved_bits(self):
    self.assertEqual([instruction.name for instruction in decode(struct.pack('<2Q', self.BAR, self.BAR | (1 << 55)))], ['bar', 'fence'])
    for word in (self.BAR | 1, self.BAR | (1 << 32)):
      with self.subTest(word=hex(word)), self.assertRaisesRegex(ValueError, 'did not recognize'):
        decode(struct.pack('<Q', word))
    synchronization_flags = sum(1 << shift for shift in (44, 51, 52, 53, 54, 59, 60))
    for word in (self.BAR, self.BAR | (1 << 55)):
      legacy = decode(struct.pack('<Q', word | synchronization_flags))[0]
      canonical = decode(struct.pack('<Q', word | synchronization_flags | (1 << 49)))[0]
      self.assertEqual(legacy.name, canonical.name)
      self.assertEqual(legacy.fields, canonical.fields)
      self.assertEqual(legacy.field('JP'), 1)

  def test_legacy_fence_preserves_jump_point_behavior(self):
    predt, prede = 0x0682000000000000, 0x0782000000000000
    write_predicate = 0x202cc0f800000000  # mov.u32u32 p0.x, c0.x
    write_output = 0x202cc00300000001  # mov.u32u32 r0.w, c0.y
    fence = self.BAR | (1 << 55) | (1 << 59)
    def execute(encoded_fence:int):
      program = struct.pack('<6Q', predt, write_predicate, encoded_fence, write_output, prede, self.END)
      return run_scalar(program, (0, 99), Memory(()), {248:1, 3:37}).regs[3]
    canonical = execute(fence | (1 << 49))
    self.assertEqual(canonical, 37)
    self.assertEqual(execute(fence), canonical)

  def test_early_exit_before_barrier_is_rejected(self):
    # br +2 skips the barrier when p0.x is true, leaving the other work-item waiting forever on hardware.
    program = struct.pack('<3Q', (1 << 55) | 2, self.BAR, self.END)
    with self.assertRaisesRegex(RuntimeError, 'exited before the workgroup barrier'):
      run_workgroup(program, (), Memory(()), [{248:1}, {248:0}], 0, 0)

  def test_different_barriers_are_rejected(self):
    program = struct.pack('<4Q', (1 << 55) | 2, self.BAR, self.BAR, self.END)
    with self.assertRaisesRegex(RuntimeError, 'different barrier instructions'):
      run_workgroup(program, (), Memory(()), [{248:1}, {248:0}], 0, 0)

  def test_uniform_shared_register_preloads(self):
    for lane_count in (65, 128):
      with self.subTest(lane_count=lane_count):
        run_workgroup(struct.pack('<Q', self.END), (), Memory(()), [{0:lane} for lane in range(lane_count)], 0, 0,
                      shared_registers={48*4:7})
    with self.assertRaisesRegex(ValueError, 'non-shared register'):
      run_workgroup(struct.pack('<Q', self.END), (), Memory(()), [{}], 0, 0, shared_registers={48*4-1:7})
    with self.assertRaisesRegex(ValueError, 'shared register write'):
      run_workgroup(struct.pack('<Q', self.END), (), Memory(()), [{48*4:7}], 0, 0)

  def test_shared_register_preloads_are_one_owned_snapshot(self):
    from test.mockgpu.qcom import emu
    source = {48*4:0x133441122}
    threads = []
    original_thread = emu.Thread
    def create_thread(*args, **kwargs):
      threads.append(thread:=original_thread(*args, **kwargs))
      return thread
    with patch.object(emu, 'Thread', side_effect=create_thread):
      run_workgroup(struct.pack('<Q', self.END), (), Memory(()), [{}, {}], 0, 0,
                    shared_registers=MappingProxyType(source))
    source[48*4] = 7
    self.assertEqual(len(threads), 2)
    self.assertIs(threads[0].shared_registers, threads[1].shared_registers)
    self.assertEqual(threads[0].read_register(48*4), 0x33441122)
    with self.assertRaises(TypeError): threads[0].shared_registers[48*4] = 9  # type: ignore[index]

  def test_compiled_shared_registers_fail_closed(self):
    output = (ctypes.c_uint32*2)()
    address = ctypes.addressof(output)
    renderer = IR3Renderer(Target('QCOM', 'IR3', 'a630'))
    renderer.prerender([])
    builder = renderer.b
    builder.shader.contents.info.workgroup_size[:] = (2, 1, 1)
    lane = nchannel(builder, nlid(builder), 0)
    value = nalu(builder, 'bcsel', nalu(builder, 'ieq', lane, nimm(builder, 0, dtypes.uint)),
                 nimm(builder, 11, dtypes.uint), nimm(builder, 42, dtypes.uint))
    first = _nread_first(builder, value)
    offset = nalu(builder, 'imul', nalu(builder, 'u2u64', lane), nimm(builder, 4, dtypes.ulong))
    nstore(builder, AddrSpace.GLOBAL, nalu(builder, 'iadd', nimm(builder, address, dtypes.ulong), offset), first)
    variant, constants, program = self._compile_nir(renderer, b'A630 shared register rejection test')
    instructions = decode(program)
    self.assertTrue(any(instruction.category == 1 and instruction.field('DST') == 48*4 for instruction in instructions))
    self.assertTrue(any(instruction.name == 'mov' and instruction.operand('SRC').bank == 'register' and
                        not instruction.operand('SRC').relative and instruction.operand('SRC').index == 48*4 for instruction in instructions))
    self.assertNotEqual(variant.cs.local_invocation_id, 0xfc)
    registers = [{variant.cs.local_invocation_id:lane_id, variant.cs.local_invocation_id+1:0,
                  variant.cs.local_invocation_id+2:0} for lane_id in range(2)]
    with self.assertRaisesRegex(ValueError, 'shared register'):
      run_workgroup(program, constants, Memory(((address, ctypes.sizeof(output)),)), registers, 1, 1)
    self.assertEqual(list(output), [0, 0])

  def test_compiled_partial_texture_mask_keeps_channel_position(self):
    # Compile the real NIR image path. Selecting green produces isam WRMASK=2; A630 keeps it in DST+1 rather than packing it into DST.
    renderer = IR3Renderer(Target('QCOM', 'IR3', 'a630'))
    renderer.prerender([])
    builder = renderer.b
    builder.shader.contents.info.workgroup_size[:] = (1, 1, 1)
    builder.shader.contents.info.num_images = 2
    coordinate = nlid(builder)
    x, y = nchannel(builder, coordinate, 0), nchannel(builder, coordinate, 1)
    value = _nload_img(builder, nimm(builder, 0, dtypes.int), y, x, dtypes.float)
    green = nchannel(builder, value, 1)
    repeated = nalu(builder, 'vec4', green, green, green, green)
    nstore_img(builder, nimm(builder, 0, dtypes.int), y, x, repeated, dtypes.float)
    variant, constant_words, program = self._compile_nir(renderer, b'A630 partial texture mask test')
    self.assertTrue(any(ins.name == 'isam' and ins.field('WRMASK') == 2 for ins in decode(program)))
    source_pixel, output_pixel = (ctypes.c_float*4)(11, 22, 33, 44), (ctypes.c_float*4)(-1, -1, -1, -1)
    source_address, output_address = ctypes.addressof(source_pixel), ctypes.addressof(output_pixel)
    memory = Memory(((source_address, 16), (output_address, 16)))
    registers = {variant.cs.local_invocation_id+i:0 for i in range(3)} if variant.cs.local_invocation_id != 0xfc else {}
    run_workgroup(program, constant_words, memory, [registers], 1, 1,
                  (Image(source_address, 1, 1, 16, False),), (Image(output_address, 1, 1, 16, False),))
    self.assertEqual(list(output_pixel), [22, 22, 22, 22])

@unittest.skipUnless('mesa' in mesa.dll._loaded_, 'requires the optional Mesa package')
class TestA630Scalar(unittest.TestCase):
  renderer:ClassVar[IR3Renderer]
  artifact:ClassVar[TinyELF]
  program:ClassVar[bytes]
  @classmethod
  def setUpClass(cls):
    cls.renderer = IR3Renderer(Target('QCOM', 'IR3', 'a630'))
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 1)
    index = UOp.const(0)
    sink = output.index(index).store(source.index(index).load()+1.0).sink(arg=KernelInfo(name='a630_add_one'))
    cls.artifact = to_program(sink, cls.renderer).to_elf()
    cls.program = IR3Compiler.unpack_lib(cls.artifact.lib)[3]

  def arguments(self, artifact:TinyELF, addresses:tuple[int, ...]) -> tuple[int, ...]:
    # Use the real runtime's compiled metadata; this unit test does not exercise a device or command queue.
    data = QCOMProgramData(cast(QCOMDevice, SimpleNamespace(renderer=self.renderer)), artifact)
    args = bytearray(data.kernargs_alloc_size)
    for value,offset,size in data.consts_info: struct.pack_into('<I' if size == 4 else '<H', args, offset, value)
    for index,address in enumerate(addresses): struct.pack_into('<Q', args, data.buf_off+index*8, address)
    return tuple(struct.unpack('<'+'I'*(len(args)//4), args))

  def test_real_compiler_instructions(self):
    names = [instruction.name for instruction in decode(self.program)]
    for required in ('mov', 'ldg', 'add.f', 'stg', 'end'): self.assertIn(required, names)

  def test_add_one_executes_compiled_a630(self):
    for value in (0.0, 1.5, -2.0, 100.0):
      with self.subTest(value=value):
        source, output = (ctypes.c_float*1)(value), (ctypes.c_float*1)(-999.0)
        out_address, in_address = ctypes.addressof(output), ctypes.addressof(source)
        constants = self.arguments(self.artifact, (out_address, in_address))
        memory = Memory(((out_address, ctypes.sizeof(output)), (in_address, ctypes.sizeof(source))))
        run_scalar(self.program, constants, memory)
        self.assertEqual(output[0], value+1.0)
        self.assertEqual(source[0], value)

  def test_scalar_arithmetic(self):
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 1)
    index = UOp.const(0)
    value = source.index(index).load()
    cases = (('multiply', value*3.0, lambda x: x*3.0),
             ('add constant', value+0.33333334, lambda x: x+ctypes.c_float(0.33333334).value),
             ('negate', -value, lambda x: -x), ('subtract', value-1.0, lambda x: x-1.0))
    for label,expression,reference in cases:
      artifact = to_program(output.index(index).store(expression).sink(arg=KernelInfo(name='a630_arithmetic')), self.renderer).to_elf()
      program = IR3Compiler.unpack_lib(artifact.lib)[3]
      for x in (0.0, -0.0, 1.5, -2.0, 100.0):
        with self.subTest(operation=label, x=x):
          inp, out = (ctypes.c_float*1)(x), (ctypes.c_float*1)(-999.0)
          addresses = (ctypes.addressof(out), ctypes.addressof(inp))
          constants = self.arguments(artifact, addresses)
          run_scalar(program, constants, Memory(tuple((address, 4) for address in addresses)))
          self.assertEqual(struct.pack('<f', out[0]), struct.pack('<f', ctypes.c_float(reference(x)).value))
          self.assertEqual(struct.pack('<f', inp[0]), struct.pack('<f', x))

  def test_compiled_vector_add(self):
    count = 8
    output, source = UOp.param(0, dtypes.float, count), UOp.param(1, dtypes.float, count)
    index = UOp.range(count, 0)
    sink = output.index(index).store(source.index(index).load()+1.0).end(index).sink(arg=KernelInfo(name='a630_vector'))
    program = to_program(sink, self.renderer)
    artifact = program.to_elf()
    data = QCOMProgramData(cast(QCOMDevice, SimpleNamespace(renderer=self.renderer)), artifact)
    self.assertEqual(program.arg.global_size, (1, 1, 1))
    self.assertEqual(program.arg.local_size[1:], (1, 1))
    inp, out = (ctypes.c_float*count)(*range(count)), (ctypes.c_float*count)(*([-999.0]*count))
    addresses = ctypes.addressof(out), ctypes.addressof(inp)
    constants = self.arguments(artifact, addresses)
    memory = Memory(tuple((address, count*4) for address in addresses))
    for local_x in range(program.arg.local_size[0]):
      run_scalar(data.image, constants, memory, {data.lid: local_x, data.lid+1: 0, data.lid+2: 0})
    self.assertEqual(list(out), [float(i+1) for i in range(count)])
    self.assertEqual(list(inp), [float(i) for i in range(count)])

  def test_a630_mad_rounds_product_before_addition(self):
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 3)
    a,b,c = [source.index(UOp.const(i)).load() for i in range(3)]
    sink = output.index(UOp.const(0)).store(a*b+c).sink(arg=KernelInfo(name='a630_mad'))
    artifact = to_program(sink, self.renderer).to_elf()
    program = IR3Compiler.unpack_lib(artifact.lib)[3]
    self.assertIn('mad.f32', [instruction.name for instruction in decode(program)])
    # Unfused rounds (1+2^-23)*(1-2^-23) to 1 before subtracting 1; fused would leave -2^-46.
    inp, out = (ctypes.c_float*3)(1+2**-23, 1-2**-23, -1), (ctypes.c_float*1)(-999)
    addresses = ctypes.addressof(out), ctypes.addressof(inp)
    run_scalar(program, self.arguments(artifact, addresses), Memory(((addresses[0], 4), (addresses[1], 12))))
    self.assertEqual(struct.pack('<f', out[0]), b'\x00\x00\x00\x00')

  def test_predication_gates_writes(self):
    # Mesa IR3 category-0 encodings plus the MOV encoding emitted by the compiler smoke kernel.
    mov = 0x202cc00300000000  # mov.u32u32 r0.w, c0.x
    prede, end = 0x0782000000000000, 0x0300000000000000
    for instruction,mode in ((0x0682000000000000, True), (0x0702000000000000, False)):
      for predicate in (0, 1):
        with self.subTest(mode=mode, predicate=predicate):
          program = struct.pack('<4Q', instruction, mov, prede, end)
          thread = run_scalar(program, (99,), Memory(()), {248: predicate, 3: 37})
          self.assertEqual(thread.regs[3], 99 if bool(predicate) == mode else 37)

  def test_jump_point_refreshes_predication_mask(self):
    predt, prede, end = 0x0682000000000000, 0x0782000000000000, 0x0300000000000000
    write_predicate = 0x202cc0f800000000  # mov.u32u32 p0.x, c0.x
    mov = 0x202cc00300000001  # mov.u32u32 r0.w, c0.y
    for jump_point in (False, True):
      with self.subTest(jump_point=jump_point):
        program = struct.pack('<5Q', predt, write_predicate, mov | (int(jump_point) << 59), prede, end)
        thread = run_scalar(program, (0, 99), Memory(()), {248: 1, 3: 37})
        self.assertEqual(thread.regs[3], 37 if jump_point else 99)

  def test_shared_register_operands_fail_closed(self):
    # r48-r55 are wave-shared on A630; full and half operands alias the same shared register file.
    instructions = (0x200cc0c000000003, 0x200cd008000000c0,  # mov r48.x, r0.w; mov r2.x, r48.x
                    0x200880c000000003, 0x20089008000000c0)  # mov hr48.x, hr0.w; mov hr2.x, hr48.x
    for instruction in instructions:
      with self.subTest(instruction=hex(instruction)), self.assertRaisesRegex(ValueError, 'shared register'):
        run_scalar(struct.pack('<2Q', instruction, 0x0300000000000000), (), Memory(()), {3: 11})

    # Repeats and relative addressing validate the effective component, not only the encoded base.
    with self.assertRaisesRegex(ValueError, 'shared register'):
      run_scalar(struct.pack('<2Q', 0x200cc900000000bf, 0x0300000000000000), (), Memory(()), {191: 11})
    relative = run_thread(struct.pack('<2Q', 0x200cc00800000800, 0x0300000000000000), (), Memory(()),
                          initial_half_registers={244:48*4})
    with self.assertRaisesRegex(ValueError, 'shared register'): next(relative)

    # Constant c48.x uses the same numeric component index but is not a shared GPR.
    result = run_scalar(struct.pack('<2Q', 0x202cc000000000c0, 0x0300000000000000), tuple(range(193)), Memory(()))
    self.assertEqual(result.regs[0], 192)

  def test_read_only_full_shared_register_preload(self):
    read = run_thread(struct.pack('<2Q', 0x200cd008000000c0, 0x0300000000000000), (), Memory(()),
                      shared_registers={48*4:0x133441122})
    with self.assertRaises(StopIteration) as completed: next(read)
    self.assertEqual(completed.exception.value.regs[8], 0x33441122)

    for instruction in (0x200cc0c000000003, 0x200880c000000003):
      with self.subTest(instruction=hex(instruction)):
        write = run_thread(struct.pack('<2Q', instruction, 0x0300000000000000), (), Memory(()), {3:11},
                           shared_registers={48*4:11})
        with self.assertRaisesRegex(ValueError, 'shared register write'): next(write)
    relative_write = run_thread(struct.pack('<2Q', 0x200ec00000000003, 0x0300000000000000), (), Memory(()), {3:11},
                                initial_half_registers={244:48*4}, shared_registers={48*4:11})
    with self.assertRaisesRegex(ValueError, 'shared register write'): next(relative_write)
    repeated_read = run_thread(struct.pack('<2Q', 0x200cc900000000bf, 0x0300000000000000), (), Memory(()), {191:10},
                               shared_registers={48*4:11})
    with self.assertRaises(StopIteration) as repeated: next(repeated_read)
    self.assertEqual(repeated.exception.value.regs[:2], [10, 11])
    half_read = run_thread(struct.pack('<2Q', 0x20089008000000c0, 0x0300000000000000), (), Memory(()),
                           shared_registers={48*4:0x33441122})
    with self.assertRaisesRegex(ValueError, 'half shared register'): next(half_read)

    pixel = (ctypes.c_float*4)(11, 22, 33, 44)
    image = Image(ctypes.addressof(pixel), 1, 1, 16, False)
    for coordinate_half in (True, False):
      instruction = (5 << 61) | (1 << 44) | (15 << 40) | (1 << 18) | (48*4 << 1) | int(not coordinate_half)
      texture = run_thread(struct.pack('<2Q', instruction, 0x0300000000000000), (),
                           Memory(((ctypes.addressof(pixel), ctypes.sizeof(pixel)),)), textures=(image,),
                           shared_registers={48*4:0, 48*4+1:0})
      if coordinate_half:
        with self.assertRaisesRegex(ValueError, 'half shared register'): next(texture)
      else:
        with self.assertRaises(StopIteration): next(texture)

  def test_signed_inline_immediate(self):
    # Compiled add.u r2.z, r0.x, -1. Mesa exposes the raw 11-bit immediate as 2047.
    program = struct.pack('<2Q', 0x4210000a27ff0000, 0x0300000000000000)
    for value in (0, 1, 8):
      with self.subTest(value=value):
        thread = run_scalar(program, (), Memory(()), {0: value})
        self.assertEqual(thread.regs[10], (value-1) & 0xffffffff)

  def test_integer_execution_modifiers(self):
    end = 0x0300000000000000
    saturating_add, halving_add = 0x4210040a00010000, 0x4210800a00010000
    for instruction,left,right,expected in ((saturating_add, 0xffffffff, 2, 0xffffffff), (saturating_add, 3, 4, 7),
                                            (halving_add, 0xffffffff, 2, 0x80000000), (halving_add, 3, 4, 3)):
      with self.subTest(instruction=hex(instruction), left=left, right=right):
        result = run_scalar(struct.pack('<2Q', instruction, end), (), Memory(()), {0:left, 1:right})
        self.assertEqual(result.regs[10], expected)
    # Only compiler-proven, full-width ADD.U uses EI as a halving add. Other forms must fail closed.
    with self.assertRaisesRegex(ValueError, 'EI modifier'):
      run_scalar(struct.pack('<2Q', 0x5230800000010000, end), (), Memory(()), {0:1, 1:2})

  def test_parallel_register_moves(self):
    # Category-1 multi-move encodings, Mesa ir3-cat1.xml: all sources are read before any destination is written.
    base, end = (1 << 61) | (2 << 57) | (3 << 50) | (3 << 46), 0x0300000000000000
    cases = ((base | (1 << 16) | (0 << 8) | 1, [20,10,30,40]),
             (base | (1 << 40) | (0 << 24) | (1 << 16) | (2 << 8) | 3, [40,30,20,10]),
             (base | (2 << 40) | (0 << 24) | (1 << 16) | (2 << 8) | (3 << 32), [40,30,20,10]))
    for instruction,expected in cases:
      with self.subTest(instruction=hex(instruction)):
        thread = run_scalar(struct.pack('<2Q', instruction, end), (), Memory(()), {0:10, 1:20, 2:30, 3:40})
        self.assertEqual(thread.regs[:4], expected)

  def test_call_and_return(self):
    # Start at a call whose helper precedes the kernel, as in vendor-compiled OpenCL math libraries.
    mov, ret, call, end = 0x202cc00000000000, 0x0200000000000000, 0x01800000fffffffe, 0x0300000000000000
    worker = run_thread(struct.pack('<4Q', mov, ret, call, end), (19,), Memory(()), start=2)
    with self.assertRaises(StopIteration) as finished: next(worker)
    self.assertEqual(finished.exception.value.regs[0], 19)

  def test_relative_constant_address(self):
    # mova a0.x, 1; mov.s32s32 r1.w, c<a0.x+96>; end, as emitted by the OpenCL math library.
    program = struct.pack('<3Q', 0x205100f400000001, 0x2015600700000c60, 0x0300000000000000)
    constants = [0]*128
    constants[96], constants[97] = 123, 456
    thread = run_scalar(program, tuple(constants), Memory(()), {96:789})
    self.assertEqual(thread.regs[7], 456)

  def test_zero_offset_relative_sources(self):
    mova, end = 0x205100f400000001, 0x0300000000000000
    # Mesa omits OFFSET=0 for both forms; the raw relative-mode bits remain authoritative.
    relative_register = 0x200cc00800000800  # mov.u32u32 r2.x, r<a0.x>
    thread = run_scalar(struct.pack('<3Q', mova, relative_register, end), (), Memory(()), {0:11, 1:22})
    self.assertEqual(thread.regs[8], 22)
    relative_constant = 0x4210000800010c00  # add.u r2.x, c<a0.x>, r0.y
    thread = run_scalar(struct.pack('<3Q', mova, relative_constant, end), (100, 200), Memory(()), {0:11, 1:22})
    self.assertEqual(thread.regs[8], 222)

  def test_signed_24bit_multiply_sign_extends_half_sources(self):
    program = struct.pack('<2Q', 0x4620400800010000, 0x0300000000000000)  # mul.s24 r2.x, hr0.x, hr0.y; end
    worker = run_thread(program, (), Memory(()), initial_half_registers={0:0xfffe, 1:3})
    with self.assertRaises(StopIteration) as finished: next(worker)
    self.assertEqual(finished.exception.value.regs[8], 0xfffffffa)

  def test_execution_budgets(self):
    end = 0x0300000000000000
    with patch.dict(os.environ, {'MOCK_QCOM_MAX_STEPS':'3'}):
      with self.assertRaisesRegex(RuntimeError, 'instruction budget'):
        run_scalar(struct.pack('<2Q', 0x0100000000000000, end), (), Memory(()))  # jump 0
    with patch.dict(os.environ, {'MOCK_QCOM_MAX_CALL_DEPTH':'2'}):
      with self.assertRaisesRegex(RuntimeError, 'call stack budget'):
        run_scalar(struct.pack('<2Q', 0x0180000000000000, end), (), Memory(()))  # call 0

  def test_half_constant_modes(self):
    program = struct.pack('<2Q', 0x2020000000000001, 0x0300000000000000)  # mov.f16f16 hr0.x, hc0.y; end
    for demotion,expected in ((False, 0x3800), (True, 0x3c00)):
      with self.subTest(demotion=demotion):
        worker = run_thread(program, (0x38000000, 0x3f800000), Memory(()), constant_demotion=demotion)
        with self.assertRaises(StopIteration) as finished: next(worker)
        self.assertEqual(finished.exception.value.half_regs[0], expected)

  def test_byte_conversion_sign_extends(self):
    # mov.u8u8 hr0.y, 253; cov.u8s16 hr0.x, hr0.y; end. The COV instruction sign-extends despite the U8 name.
    program = struct.pack('<3Q', 0x20598001000000fd, 0x2019000000000001, 0x0300000000000000)
    self.assertEqual(run_scalar(program, (), Memory(())).half_regs[0], 0xfffd)

  def test_inverted_comparison(self):
    # Qualcomm emits bit 42 for !(a < b), including the unordered/NaN case. Mesa displays this comparison flag as (sat).
    program = struct.pack('<2Q', 0x40b004f800010000, 0x0300000000000000)
    for value in (-2.0, 0.0, 1.0, 2.0, float('nan'), float('inf')):
      with self.subTest(value=value):
        thread = run_scalar(program, (), Memory(()), {0:f32bits(value), 1:f32bits(1.0)})
        self.assertEqual(thread.regs[248], int(not value < 1.0))

  def test_clz_zero_sentinel(self):
    program = struct.pack('<2Q', 0x46b0000100000000, 0x0300000000000000)  # clz.b r0.y, r0.x; end
    for value,expected in ((0, 0xffffffff), (1, 31), (0x80000000, 0), (0xffffffff, 0)):
      with self.subTest(value=value):
        self.assertEqual(run_scalar(program, (), Memory(()), {0:value}).regs[1], expected)

  def test_mad_u16_full_register_form(self):
    # Emitted for (uint_a & 65535) * (uint_b & 65535) + uint_c; Mesa 25 labels these full-GPR sources as half.
    program = struct.pack('<2Q', 0x7003400200030002, 0x0300000000000000)
    for a,b,c in ((3,4,5), (0x12345678,0xabcd9876,0x34567890)):
      with self.subTest(a=a, b=b, c=c):
        result = run_scalar(program, (), Memory(()), {2:a, 6:b, 3:c})
        self.assertEqual(result.regs[2], ((a & 65535)*(b & 65535)+c) & 0xffffffff)

  def test_negative_private_store_offset(self):
    # Compiled stp.f32 p[r45.w-60], r0.x, 1. Mesa's callback returns raw 13-bit offset 8132.
    scratch = ctypes.create_string_buffer(64)
    pointer = ctypes.addressof(scratch)
    program = struct.pack('<2Q', 13926097360389160448, 0x0300000000000000)
    worker = run_thread(program, (), Memory(((pointer, 64),)), {183:60, 0:f32bits(3.5)}, private=pointer, private_size=64)
    with self.assertRaises(StopIteration): next(worker)
    self.assertEqual(ctypes.c_float.from_buffer(scratch).value, 3.5)
    self.assertEqual(scratch.raw[4:], bytes(60))

  def test_local_and_private_accesses_cannot_escape_into_global_memory(self):
    backing = ctypes.create_string_buffer(8192)
    local_private, global_address = ctypes.addressof(backing), ctypes.addressof(backing)+4096
    sentinel = ctypes.c_uint32.from_address(global_address)
    sentinel.value = 0x1234abcd
    memory = Memory(((local_private, 16), (global_address, 16)))
    end = 0x0300000000000000
    cases = (
      ('local load', 0xC046001401840001, {16:4096}, {'shared':local_private, 'shared_size':16}),
      ('local store', 0xC106110001800018, {8:4096, 12:0xdeadbeef}, {'shared':local_private, 'shared_size':16}),
      ('private load', 0xC08600000180C021, {3:4080}, {'private':local_private, 'private_size':16}),
      ('private store', 0xC146071001800000, {3:4080, 0:0xdeadbeef}, {'private':local_private, 'private_size':16}),
    )
    for label,instruction,registers,spaces in cases:
      with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'access outside allocated memory'):
        next(run_thread(struct.pack('<2Q', instruction, end), (), memory, registers, **spaces))
      self.assertEqual(sentinel.value, 0x1234abcd)

  def test_local_access_validates_the_complete_vector(self):
    backing = ctypes.create_string_buffer(32)
    local, adjacent_global = ctypes.addressof(backing), ctypes.addressof(backing)+16
    ctypes.c_uint32.from_address(local+12).value = 0x11111111
    ctypes.c_uint32.from_address(adjacent_global).value = 0x22222222
    memory = Memory(((local, 16), (adjacent_global, 16)))
    end = 0x0300000000000000
    last_word = run_thread(struct.pack('<2Q', 0xC046001401840001, end), (), memory, {16:12}, shared=local, shared_size=16)
    with self.assertRaises(StopIteration) as completed: next(last_word)
    self.assertEqual(completed.exception.value.regs[20], 0x11111111)

    for instruction,registers in ((0xC046001402840001, {16:12}), (0xC106110002800018, {8:12, 12:3, 13:4})):
      with self.subTest(instruction=hex(instruction)), patch.object(memory, 'read', wraps=memory.read) as read, \
           patch.object(memory, 'write', wraps=memory.write) as write, self.assertRaisesRegex(ValueError, 'access outside allocated memory'):
        next(run_thread(struct.pack('<2Q', instruction, end), (), memory, registers, shared=local, shared_size=16))
      read.assert_not_called()
      write.assert_not_called()
      self.assertEqual(ctypes.c_uint32.from_address(local+12).value, 0x11111111)
      self.assertEqual(ctypes.c_uint32.from_address(adjacent_global).value, 0x22222222)

  def test_integer_to_float_rounding(self):
    # COV's default mode is toward zero; mode 1 is nearest-even. These integers lie exactly between adjacent float32 values.
    for value,expected in ((16777219, (16777218,16777220,16777220,16777218)),
                           (-16777219, (-16777218,-16777220,-16777218,-16777220))):
      for rounding,target in enumerate(expected):
        with self.subTest(value=value, rounding=rounding):
          program = struct.pack('<2Q', 0x2014400100000000 | (rounding << 55), 0x0300000000000000)
          self.assertEqual(run_scalar(program, (), Memory(()), {0:value}).regs[1], f32bits(target))

  def test_float_to_half_rounding(self):
    # COV.f32f16 must honor the encoded mode, not silently turn ROUND_ZERO into nearest-even.
    for value,expected in ((1+3*2**-12, (0x3c00,0x3c01,0x3c01,0x3c00)),
                           (-1-3*2**-12, (0xbc00,0xbc01,0xbc00,0xbc01))):
      for rounding,target in enumerate(expected):
        with self.subTest(value=value, rounding=rounding):
          program = struct.pack('<2Q', 0x2004000100000000 | (rounding << 55), 0x0300000000000000)
          self.assertEqual(run_scalar(program, (), Memory(()), {0:f32bits(value)}).half_regs[1], target)

  def test_signed_selection_includes_zero(self):
    # Vendor-compiled indexing uses SEL.S32 to implement index >= 0 ? index : index+length.
    program = struct.pack('<2Q', 0x6580800300020000, 0x0300000000000000)
    for selector,expected in ((-2, 9), (0, 7), (2, 7)):
      with self.subTest(selector=selector):
        self.assertEqual(run_scalar(program, (), Memory(()), {0:7, 1:selector, 2:9}).regs[3], expected)

if __name__ == '__main__': unittest.main()
