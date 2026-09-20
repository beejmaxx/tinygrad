import dataclasses, itertools, math, struct, time
from collections import defaultdict
from collections.abc import Callable
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.emu import Image, Memory, SHARED_REGISTER_BASE, SHARED_REGISTER_END, decode, run_workgroup

def field(word:int, name:str) -> int: return (word & getattr(mesa, name+'__MASK')) >> getattr(mesa, name+'__SHIFT')
def address(words:list[int], offset:int=0) -> int: return words[offset] | words[offset+1] << 32
def parity(value:int) -> int:
  for shift in range(4, 1, -1): value ^= value >> (1 << shift)
  return (~0x6996 >> (value & 0xf)) & 1

class PendingWait(RuntimeError): pass

@dataclasses.dataclass(frozen=True)
class Packet:
  kind:int
  target:int
  values:tuple[int, ...]

@dataclasses.dataclass
class GPUState:
  registers:dict[int, int] = dataclasses.field(default_factory=dict)
  constants:list[int] = dataclasses.field(default_factory=lambda: [0]*1024)
  program_address:int = 0
  program_size:int = 0
  state_blocks:dict[tuple[int, int, int], tuple[int, ...]] = dataclasses.field(default_factory=dict)

class QCOMGPU:
  def __init__(self, ranges:Callable[[], tuple[tuple[int, int], ...]]):
    self.ranges, self.dispatches = ranges, 0
    self.load_state(GPUState())

  def load_state(self, state:GPUState):
    self.registers, self.constants = defaultdict[int, int](int, state.registers), state.constants.copy()
    self.program_address, self.program_size, self.state_blocks = state.program_address, state.program_size, state.state_blocks.copy()

  def save_state(self) -> GPUState:
    return GPUState(dict(self.registers), self.constants.copy(), self.program_address, self.program_size, self.state_blocks.copy())

  def capture(self, pointer:int, size:int, ranges:tuple[tuple[int, int], ...]|None=None) -> tuple[Packet, ...]:
    memory = Memory(self.ranges() if ranges is None else ranges)
    if size == 0 or size % 4: raise ValueError(f'A630 command buffer has invalid size: {size}')
    memory.check(pointer, size)
    words = list(struct.unpack('<'+'I'*(size//4), memory.read_bytes(pointer, size)))
    packets:list[Packet] = []
    cursor = 0
    while cursor < len(words):
      header = words[cursor]
      kind = header >> 28
      if kind == 4:
        count, register = header & 0x7f, (header >> 8) & 0x3ffff
        expected = mesa.CP_TYPE4_PKT | count | parity(count) << 7 | register << 8 | parity(register) << 27
        if header != expected: raise ValueError(f'Invalid A630 type-4 packet header at dword {cursor}')
        if count == 0 or register+count > 0x40000: raise ValueError(f'A630 type-4 packet has invalid extent at dword {cursor}')
        target = register
      elif kind == 7:
        count, opcode = header & 0x3fff, (header >> 16) & 0x7f
        expected = mesa.CP_TYPE7_PKT | count | parity(count) << 15 | opcode << 16 | parity(opcode) << 23
        if header != expected: raise ValueError(f'Invalid A630 type-7 packet header at dword {cursor}')
        target = opcode
      else: raise ValueError(f'Unsupported A630 packet type {kind}')
      values = words[cursor+1:cursor+1+count]
      if len(values) != count: raise ValueError('Truncated A630 command packet')
      packets.append(Packet(kind, target, tuple(values)))
      cursor += count+1
    return tuple(packets)

  def execute_packets(self, packets:tuple[Packet, ...], start:int=0,
                      ranges:tuple[tuple[int, int], ...]|None=None) -> tuple[bool, int]:
    if not 0 <= start <= len(packets): raise ValueError(f'Invalid A630 command cursor {start}')
    memory = Memory(self.ranges() if ranges is None else ranges, transactional=True)
    saved = (self.registers.copy(), self.constants.copy(), self.program_address, self.program_size,
             self.state_blocks.copy(), self.dispatches)
    try:
      for index in range(start, len(packets)):
        packet = packets[index]
        if packet.kind == 4: self.registers.update((packet.target+i, value) for i,value in enumerate(packet.values))
        else:
          try: self.packet(packet.target, list(packet.values), memory)
          except PendingWait:
            memory.commit()
            return False,index
      memory.commit()
    except Exception:
      self.registers, self.constants, self.program_address, self.program_size, self.state_blocks, self.dispatches = saved
      raise
    return True,len(packets)

  def submit(self, pointer:int, size:int):
    complete,_ = self.execute_packets(self.capture(pointer, size))
    if not complete: raise PendingWait('A630 command is waiting on an unsatisfied memory dependency')

  def packet(self, opcode:int, values:list[int], memory:Memory):
    exact_counts = {mesa.CP_WAIT_FOR_IDLE:0, mesa.CP_WAIT_MEM_WRITES:0, mesa.CP_SET_MARKER:1, mesa.CP_REG_TO_MEM:3,
                    mesa.CP_WAIT_REG_MEM:6, mesa.CP_LOAD_STATE6_FRAG:3, mesa.CP_EXEC_CS:4, mesa.CP_RUN_OPENCL:1}
    if opcode in exact_counts and len(values) != exact_counts[opcode]:
      raise ValueError(f'A630 command {opcode:#x} expects {exact_counts[opcode]} dwords, got {len(values)}')
    if opcode in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES): return
    if opcode == mesa.CP_SET_MARKER:
      if values != [mesa.RM6_COMPUTE]: raise ValueError(f'Unsupported A630 marker {values[0]:#x}')
      return
    if opcode == mesa.CP_EVENT_WRITE:
      if not values: raise ValueError('A630 event write is missing its event')
      event = field(values[0], 'CP_EVENT_WRITE_0_EVENT')
      expected = 4 if event == mesa.CACHE_FLUSH_TS else 1 if event == mesa.CACHE_INVALIDATE else None
      if expected is None: raise ValueError(f'Unsupported A630 event {event}')
      if len(values) != expected: raise ValueError(f'A630 event {event} expects {expected} dwords, got {len(values)}')
      if event == mesa.CACHE_FLUSH_TS: memory.write(address(values, 1), 4, values[3])
    elif opcode == mesa.CP_REG_TO_MEM:
      control = values[0]
      if (field(control, 'CP_REG_TO_MEM_0_REG') != mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER or
          field(control, 'CP_REG_TO_MEM_0_CNT') != 2 or not control & mesa.CP_REG_TO_MEM_0_64B or control & mesa.CP_REG_TO_MEM_0_ACCUMULATE):
        raise ValueError('Unsupported A630 register-to-memory operation')
      memory.write(address(values, 1), 8, time.perf_counter_ns()*192//10000)
    elif opcode == mesa.CP_WAIT_REG_MEM:
      if field(values[0], 'CP_WAIT_REG_MEM_0_POLL') != mesa.POLL_MEMORY: raise ValueError('Unsupported A630 register wait')
      if values[0] & (mesa.CP_WAIT_REG_MEM_0_SIGNED_COMPARE | mesa.CP_WAIT_REG_MEM_0_WRITE_MEMORY):
        raise ValueError('Unsupported A630 memory wait mode')
      actual, reference = memory.read(address(values, 1), 4) & values[4], values[3] & values[4]
      function = field(values[0], 'CP_WAIT_REG_MEM_0_FUNCTION')
      satisfied = {mesa.WRITE_GE:actual >= reference, mesa.WRITE_EQ:actual == reference, mesa.WRITE_NE:actual != reference,
                   mesa.WRITE_ALWAYS:True}.get(function)
      if satisfied is None: raise ValueError(f'Unsupported A630 wait function {function}')
      if not satisfied: raise PendingWait('A630 command is waiting on an unsatisfied memory dependency')
    elif opcode == mesa.CP_LOAD_STATE6_FRAG:
      state = values[0]
      block, kind, units = (field(state, 'CP_LOAD_STATE6_0_'+name) for name in ('STATE_BLOCK', 'STATE_TYPE', 'NUM_UNIT'))
      if field(state, 'CP_LOAD_STATE6_0_STATE_SRC') != mesa.SS6_INDIRECT: raise ValueError('Unsupported direct A630 state load')
      if units == 0: raise ValueError('A630 state load must contain at least one unit')
      pointer = address(values, 1)
      if block == mesa.SB6_CS_SHADER and kind == mesa.ST_CONSTANTS:
        offset = field(state, 'CP_LOAD_STATE6_0_DST_OFF')*4
        if offset+units*4 > len(self.constants): raise ValueError('A630 constant load exceeds constant RAM')
        memory.check(pointer, units*16)
        self.constants[offset:offset+units*4] = struct.unpack('<'+'I'*(units*4), memory.read_bytes(pointer, units*16))
      elif block == mesa.SB6_CS_SHADER and kind == mesa.ST_SHADER:
        self.program_address, self.program_size = pointer, units*128
      elif (block,kind) in ((mesa.SB6_CS_SHADER,mesa.ST6_UAV), (mesa.SB6_CS_TEX,mesa.ST_CONSTANTS), (mesa.SB6_CS_TEX,mesa.ST_SHADER)):
        stride = 16 if kind == mesa.ST_SHADER else 64
        memory.check(pointer, units*stride)
        offset = field(state, 'CP_LOAD_STATE6_0_DST_OFF')
        for i in range(units):
          self.state_blocks[block,kind,offset+i] = struct.unpack('<'+'I'*(stride//4), memory.read_bytes(pointer+i*stride, stride))
      else: raise ValueError(f'Unsupported A630 state block/type {block}/{kind}')
    elif opcode == mesa.CP_EXEC_CS:
      if values[0] != 0: raise ValueError(f'Unsupported A630 execute control {values[0]:#x}')
      self.execute(tuple(values[1:4]), memory)
    elif opcode == mesa.CP_RUN_OPENCL:
      if values[0] != 0: raise ValueError(f'Unsupported A630 OpenCL control {values[0]:#x}')
      self.execute(tuple(self.registers[mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X+i] for i in range(3)), memory)
    else: raise ValueError(f'Unsupported A630 command {opcode:#x}')

  def samplers(self, memory:Memory, texture_count:int) -> frozenset[int]:
    count = field(self.registers[mesa.REG_A6XX_SP_CS_CONFIG], 'A6XX_SP_CS_CONFIG_NSAMP')
    base = self.registers[mesa.REG_A6XX_SP_CS_SAMPLER_BASE] | self.registers[mesa.REG_A6XX_SP_CS_SAMPLER_BASE+1] << 32
    samplers = []
    for index in range(count):
      words = self.state_blocks.get((mesa.SB6_CS_TEX, mesa.ST_SHADER, index))
      if words is None:
        memory.check(base+index*16, 16)
        words = struct.unpack('<4I', memory.read_bytes(base+index*16, 16))
      samplers.append(words)
    while samplers and samplers[-1] == (0, 0, 0, 0): samplers.pop()
    if texture_count and not samplers: raise ValueError('A630 texture configuration is missing an active sampler')
    expected = (3 << 5 | 3 << 8 | 3 << 11, 0x30, 0, 0)
    if any(words != expected for words in samplers): raise ValueError('Unsupported A630 image sampler')
    if samplers:
      border = (self.registers[mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE] |
                self.registers[mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE+1] << 32)
      if border == 0 or border % 64: raise ValueError('Invalid A630 image border color base')
      memory.check(border, 4096)
      if any(memory.read_bytes(border, 4096)): raise ValueError('Unsupported nonzero A630 image border color')
    return frozenset(range(len(samplers)))

  def images(self, memory:Memory, textures:bool) -> tuple[Image, ...]:
    count = field(self.registers[mesa.REG_A6XX_SP_CS_CONFIG], 'A6XX_SP_CS_CONFIG_'+('NTEX' if textures else 'NUAV'))
    block, kind = (mesa.SB6_CS_TEX, mesa.ST_CONSTANTS) if textures else (mesa.SB6_CS_SHADER, mesa.ST6_UAV)
    base = mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE if textures else mesa.REG_A6XX_SP_CS_UAV_BASE
    pointer = self.registers[base] | self.registers[base+1] << 32
    result = []
    for index in range(count):
      words = self.state_blocks.get((block,kind,index))
      if words is None:
        memory.check(pointer+index*64, 64)
        words = struct.unpack('<16I', memory.read_bytes(pointer+index*64, 64))
      fmt = field(words[0], 'A6XX_TEX_CONST_0_FMT')
      if fmt not in (mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT): raise ValueError(f'Unsupported A630 image format {fmt}')
      expected_swizzle = (1 << 7) | (2 << 10) | (3 << 13) | 8 if textures else 0
      if words[0] != fmt << mesa.A6XX_TEX_CONST_0_FMT__SHIFT | expected_swizzle or words[1] & ~0x3fffffff:
        raise ValueError('Unsupported A630 image layout or swizzle')
      width, height = field(words[1], 'A6XX_TEX_CONST_1_WIDTH'), field(words[1], 'A6XX_TEX_CONST_1_HEIGHT')
      pitch = field(words[2], 'A6XX_TEX_CONST_2_PITCH')
      pitch_alignment = (pitch & -pitch).bit_length()-7
      expected_pitch = mesa.A6XX_TEX_2D << mesa.A6XX_TEX_CONST_2_TYPE__SHIFT | pitch << mesa.A6XX_TEX_CONST_2_PITCH__SHIFT | pitch_alignment
      if pitch_alignment < 0 or words[2] != expected_pitch: raise ValueError('Unsupported A630 image pitch or texture type')
      if words[3] or words[4] & 31 or words[5] & ~0x1ffff: raise ValueError('Unsupported A630 image layers or base address')
      if words[6:] != (0x40000000, 13, 0, 0, 0, 0, 0, 0, 0, 0):
        raise ValueError('Unsupported A630 image planes or compression')
      image_pointer, component_size = words[4] | words[5] << 32, 2 if fmt == mesa.FMT6_16_16_16_16_FLOAT else 4
      if width == 0 or height == 0 or image_pointer == 0 or pitch < width*4*component_size or pitch % 64:
        raise ValueError('Invalid A630 image dimensions or row pitch')
      memory.check(image_pointer, (height-1)*pitch+width*4*component_size)
      result.append(Image(image_pointer, width, height, pitch, component_size == 2))
    return tuple(result)

  @staticmethod
  def validate_image_accesses(program:bytes, texture_count:int, image_count:int, samplers:frozenset[int]):
    for instruction in decode(program):
      if instruction.name == 'isam':
        if any(instruction.field(flag) for flag in ('3D','A','O','P','S2EN','S2EN_BINDLESS')):
          raise ValueError('Unsupported A630 texture addressing mode')
        if instruction.field('TEX') >= texture_count: raise ValueError('A630 instruction selects an unavailable texture')
        if instruction.field('SAMP') not in samplers: raise ValueError('A630 instruction selects an unavailable sampler')
      elif instruction.name in ('stib.b', 'ldib.b') and instruction.field('SSBO') >= image_count:
        raise ValueError('A630 instruction selects an unavailable image')

  def execute(self, groups:tuple[int, ...], memory:Memory):
    geometry = self.registers[mesa.REG_A6XX_SP_CS_NDRANGE_0]
    local_size = tuple(field(geometry, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZE'+axis)+1 for axis in 'XYZ')
    if any(count == 0 for count in groups): raise ValueError(f'Invalid A630 workgroup count {groups}')
    if math.prod(local_size) > 1024: raise ValueError(f'Invalid A630 local size {local_size}')
    config = self.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    wgid, wgsz, wgoff, lid = (field(config, 'A6XX_SP_CS_CONST_CONFIG_0_'+name)
                             for name in ('WGIDCONSTID', 'WGSIZECONSTID', 'WGOFFSETCONSTID', 'LOCALIDREGID'))
    linear = field(self.registers[mesa.REG_A6XX_SP_CS_WGE_CNTL], 'A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID')
    destinations:dict[int, str] = {}
    for base,width,name,lane_local in ((wgid, 3, 'workgroup ID', False), (wgsz, 3, 'workgroup size', False),
                                       (wgoff, 3, 'workgroup offset', False), (lid, 3, 'local ID', True),
                                       (linear, 1, 'linear local ID', True)):
      if base == 0xfc: continue
      if base+width > SHARED_REGISTER_END: raise ValueError(f'A630 {name} register range is outside the GPR files')
      if lane_local and base < SHARED_REGISTER_END and base+width > SHARED_REGISTER_BASE:
        raise ValueError(f'A630 {name} cannot target shared registers')
      for index in range(base, base+width):
        if index in destinations: raise ValueError(f'A630 launch metadata register {index} overlaps {destinations[index]} and {name}')
        destinations[index] = name
    start = self.registers[mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET]
    memory.check(self.program_address, self.program_size)
    # Keep the whole image: OpenCL kernels can call helper functions preceding their entry point.
    program = memory.read_bytes(self.program_address, self.program_size)
    shared_size = (field(self.registers[mesa.REG_A6XX_SP_CS_CNTL_1], 'A6XX_SP_CS_CNTL_1_SHARED_SIZE')+1)*1024
    private_size = field(self.registers[mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM], 'A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM')*512
    textures, images = self.images(memory, True), self.images(memory, False)
    samplers = self.samplers(memory, len(textures))
    self.validate_image_accesses(program, len(textures), len(images), samplers)
    constant_demotion = bool(self.registers[mesa.REG_A6XX_SP_MODE_CNTL] & mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE)
    for group in itertools.product(*(range(count) for count in groups)):
      uniform_registers,shared_registers = {}, {}
      # Despite their names, these fields select GPRs, not entries in constant RAM (Mesa a6xx.xml).
      for base,values in ((wgid, group), (wgsz, local_size), (wgoff, tuple(g*l for g,l in zip(group, local_size)))):
        if base != 0xfc:
          for i,value in enumerate(values):
            (shared_registers if base+i >= SHARED_REGISTER_BASE else uniform_registers)[base+i] = value
      workgroup = []
      for local in itertools.product(*(range(count) for count in local_size)):
        registers = uniform_registers | ({lid+i:value for i,value in enumerate(local)} if lid != 0xfc else {})
        if linear != 0xfc: registers[linear] = local[0]+local_size[0]*(local[1]+local_size[1]*local[2])
        workgroup.append(registers)
      run_workgroup(program, tuple(self.constants), memory, workgroup, shared_size, private_size, textures, images, start, constant_demotion,
                    shared_registers=shared_registers)
    self.dispatches += 1
