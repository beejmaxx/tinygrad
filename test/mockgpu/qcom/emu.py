"""A630 instruction execution for MockGPU, decoded by the existing Mesa library."""
from __future__ import annotations
import ctypes, dataclasses, functools, math, os, struct, tempfile
from collections.abc import Callable, Generator, Mapping
from types import MappingProxyType
from typing import Literal
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.host import read_host, validate_host_mapping, write_host

def f32bits(value:float) -> int: return struct.unpack('<I', struct.pack('<f', ctypes.c_float(value).value))[0]
def bitsf32(value:int) -> float: return struct.unpack('<f', struct.pack('<I', value & 0xffffffff))[0]
def f16bits(value:float) -> int:
  try: return struct.unpack('<H', struct.pack('<e', value))[0]
  except OverflowError: return 0xfc00 if value < 0 else 0x7c00
def bitsf16(value:int) -> float: return struct.unpack('<e', struct.pack('<H', value & 0xffff))[0]
def rounded_float(value:int|float, half:bool, rounding:int) -> int:
  bits = (f16bits if half else f32bits)(value)
  rounded = (bitsf16 if half else bitsf32)(bits)
  if rounding == 1 or math.isnan(value) or rounded == value: return bits
  up = rounding == 2 or (rounding == 0 and value < 0)
  if (up and rounded < value) or (not up and rounded > value):
    bits += -1 if up == bool(bits & (0x8000 if half else 0x80000000)) else 1
  return bits
def special_float(name:str, value:float) -> float:
  if name in ('floor.f', 'ceil.f', 'trunc.f', 'rndne.f'):
    if not math.isfinite(value) or value == 0: return value
    rounders:dict[str, Callable[[float], int]] = {'floor.f': math.floor, 'ceil.f': math.ceil, 'trunc.f': math.trunc, 'rndne.f': round}
    rounded = float(rounders[name](value))
    return math.copysign(rounded, value) if rounded == 0 else rounded
  if name == 'rcp': return math.copysign(math.inf, value) if value == 0 else 1/value
  if name in ('sqrt', 'rsq'):
    root = math.nan if value < 0 else math.sqrt(value)
    return root if name == 'sqrt' else math.copysign(math.inf, root) if root == 0 else 1/root
  if name == 'log2': return -math.inf if value == 0 else math.nan if value < 0 else math.log2(value)
  try: return {'exp2': math.exp2, 'sin': math.sin, 'cos': math.cos}[name](value)
  except OverflowError: return math.inf
  except ValueError: return math.nan
def signed(value:int, bits:int=32) -> int:
  value &= (1 << bits)-1
  return value-(1 << bits) if value & (1 << (bits-1)) else value

def convert(value:int, source:int, destination:int, rounding:int) -> int:
  if source == destination: return value
  widths = (16, 32, 16, 32, 16, 32, 8, 8)
  if source == 0: number:int|float = bitsf16(value)
  elif source == 1: number = bitsf32(value)
  # Despite its U8 name, COV sign-extends bytes; Mesa uses AND 0xff for zero-extension (create_cov in ir3_compiler_nir.c).
  elif source in (4, 5, 6): number = signed(value, widths[source])
  else: number = value & ((1 << widths[source])-1)
  if destination in (0, 1): return rounded_float(number, destination == 0, rounding)
  if isinstance(number, float): number = (math.trunc, round, math.ceil, math.floor)[rounding](number)
  return number & ((1 << widths[destination])-1)

class Fields:
  fields:tuple[tuple[str, int | str], ...]
  @functools.cached_property
  def first(self) -> dict[str, int|str]: return dict(reversed(self.fields))
  def field(self, name:str, default:int=0) -> int: return int(self.first.get(name, default))

@dataclasses.dataclass(frozen=True)
class Operand:
  bank:Literal['register', 'constant', 'immediate']
  index:int  # immediate bits, an absolute component index, or a signed relative offset
  half:bool
  relative:bool
  repeat:bool  # advance the source component on each instruction repeat
  absneg:int

@dataclasses.dataclass(frozen=True)
class Instruction(Fields):
  word:int
  fields:tuple[tuple[str, int | str], ...]
  @property
  def category(self) -> int: return self.word >> 61
  @functools.cached_property
  def name(self) -> str:
    if self.category == 7 and (opcode:=(self.word >> 55) & 15) in (0, 1): return ('bar', 'fence')[opcode]
    if self.category == 1:
      opcode = (self.word >> 57) & 3
      if opcode == 2: return ('swz', 'gat', 'sct', 'invalid')[(self.word >> 40) & 3]
      if opcode == 3: return 'movmsk'
      if opcode == 0: return 'movs' if (self.word >> 53) & 3 == 0 and self.word & (1 << 31) else 'mov'
    return next((str(value) for key,value in self.fields if key == 'NAME'), '')
  @functools.cached_property
  def operands(self) -> dict[str, Operand]: return {}
  @functools.cached_property
  def first(self) -> dict[str, int|str]:
    # MOVA's display asserts its types and destination, so Mesa does not emit callbacks for those fields.
    values = super().first
    if self.category == 1:
      values.update(SRC_TYPE=(self.word >> 50) & 7, DST_TYPE=(self.word >> 46) & 7, DST=(self.word >> 32) & 255,
                    DST_HALF=int(((self.word >> 46) & 7) in (0,2,4,6,7)), DST_REL=(self.word >> 49) & 1)
    return values
  def operand(self, name:str) -> Operand:
    if name in self.operands: return self.operands[name]
    index = next(i for i,(key,_) in enumerate(self.fields) if key == name)
    boundaries = {'DST', 'SRC'} if self.category == 1 else {'DST', 'SRC1', 'SRC2', 'SRC3', 'SRC4', 'SIZE', 'OFF'}
    end = next((i for i in range(index+1, len(self.fields)) if self.fields[i][0] in boundaries - {name}), len(self.fields))
    encoded = int(self.fields[index][1])
    fields = self.fields[index+1:end]
    if self.category == 1:
      # Mesa omits OFFSET when it is zero, so relative addressing must be recovered from the encoding.
      relative = name == 'SRC' and (self.word >> 53) & 3 == 0 and bool(self.word & (1 << 11))
      fields += (('HALF', int(self.field('SRC_TYPE') in (0,2,4,6,7))), ('REL', int(relative)),
                 ('REL_CONST', (self.word >> 10) & 1))
    elif self.category in (2, 4) and name.startswith('SRC'):
      encoded = (self.word >> (16 if name == 'SRC2' else 0)) & 0xffff
      fields += (('REL', int((encoded >> 11) & 7 == 1)), ('REL_CONST', (encoded >> 10) & 1))
    elif self.category == 3:
      fields += (('SRC_R', self.field(name+'_R')), ('ABSNEG', self.field(name+'_NEG')), ('HALF', self.field('HALF')))
      # Qualcomm emits this form for low-16-bit multiplication plus a full 32-bit addend, all in full GPRs.
      if self.name == 'mad.u16' and not self.field('DST_HALF'): fields = (('HALF', 0),)+fields
    first,last = dict(reversed(fields)),dict(fields)
    half = bool(first['HALF']) if 'HALF' in first else self.category in (2,4) and (encoded >> 11) & 7 == 5 and bool(encoded & (1 << 10))
    immediate = None
    if 'IMMED' in first:
      immediate = int(first['IMMED'])
      if self.category in (2, 4) and (encoded >> 11) & 7 == 5:
        if immediate >= len(FLOAT_IMMEDIATES): raise ValueError(f'Unsupported floating immediate {immediate}')
        immediate = (f16bits if half else f32bits)(FLOAT_IMMEDIATES[immediate])
      else:
        if self.category in (2, 4): immediate = signed(immediate, 11)
        immediate &= 0xffff if half else 0xffffffff
    relative = bool(first.get('REL', 0)) or 'OFFSET' in first
    if immediate is not None: bank,component = 'immediate',immediate
    elif relative: bank,component = ('constant' if first.get('REL_CONST', 0) else 'register'),signed(int(first.get('OFFSET', 0)), 10)
    elif 'CONST' in first: bank,component = 'constant',int(last['CONST'])*4+int(last.get('SWIZ', 0))
    else: bank,component = 'register',int(first.get('SRC', encoded & 255))
    self.operands[name] = Operand(bank, component, half, relative, bool(first.get('SRC_R', 0)), int(first.get('ABSNEG', 0)))
    return self.operands[name]

@functools.cache
def decode(program:bytes) -> tuple[Instruction, ...]:
  if len(program) % 8: raise ValueError('A630 instructions must contain complete 64-bit words')
  rows:list[tuple[int, list[tuple[str, int | str]]]] = []
  errors:list[Exception] = []
  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def pre(_data, _pc, instruction):
    rows.append((ctypes.cast(instruction, ctypes.POINTER(ctypes.c_uint64)).contents.value, []))
  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value))
  def field(_data, name, value):
    try:
      decoded = value.contents
      rows[-1][1].append((ctypes.string_at(name).decode(), ctypes.string_at(decoded.str).decode() if decoded.str else decoded.num))
    except Exception as error: errors.append(error)
  @ctypes.CFUNCTYPE(None, ctypes.POINTER(mesa.struct__IO_FILE), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64)
  def no_match(_output, words, count):
    try:
      # Mesa does not match legacy BAR/FENCE without its fixed bit 49. Accept only their documented flags, not arbitrary unmatched words.
      flags = (('SY', 60), ('SS', 44), ('JP', 59), ('G', 54), ('L', 53), ('R', 52), ('W', 51))
      allowed = (1 << 55) | sum(1 << shift for _,shift in flags)
      word = int(words[0]) | int(words[1]) << 32 if count == 2 else 0
      if word & ~allowed != 7 << 61: raise ValueError(f'Mesa did not recognize A630 instruction {word:#x}')
      rows[-1][1].extend(((name, (word >> shift) & 1) for name,shift in flags))
      rows[-1][1].insert(3, ('NAME', 'fence' if word & (1 << 55) else 'bar'))
    except Exception as error: errors.append(error)
  with tempfile.TemporaryFile('w+') as output:
    fd = os.dup(output.fileno())
    fp = libc.fdopen(fd, b'w')
    if not fp:
      os.close(fd)
      raise OSError('fdopen failed while decoding A630 instructions')
    try:
      options = mesa.struct_isa_decode_options(gpu_id=630, show_errors=True, branch_labels=False, pre_instr_cb=pre, field_cb=field,
                                               no_match_cb=no_match)
      mesa.ir3_isa_disasm(program, len(program), ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE)), options)
    finally: libc.fclose(fp)
  if errors: raise errors[0]
  if len(rows)*8 != len(program): raise ValueError(f'Mesa decoded {len(rows)} of {len(program)//8} A630 instructions')
  return tuple(Instruction(word, tuple(fields)) for word,fields in rows)

class Memory:
  def __init__(self, ranges:tuple[tuple[int, int], ...], transactional:bool=False, *,
               snapshots:dict[tuple[int, int], bytearray]|None=None, dirty:dict[tuple[int, int], bytearray]|None=None,
               direct:tuple[tuple[int, int], ...]=()):
    canonical:dict[int, int] = {}
    for start,length in ranges:
      if start < 0 or length < 0: raise ValueError(f'A630 memory has an invalid range: {start:#x} + {length}')
      canonical[start] = max(canonical.get(start, 0), length)
    ordered = tuple(sorted(canonical.items()))
    if any(start+length > next_start for (start,length),(next_start,_) in zip(ordered, ordered[1:])):
      raise ValueError('A630 memory ranges overlap')
    self.ranges, self.transactional = ordered, transactional
    self.snapshots = {} if snapshots is None else snapshots
    self.dirty = {} if dirty is None else dirty
    self.direct = direct
  def region(self, address:int, size:int) -> tuple[int, int]:
    if size < 0: raise ValueError(f'A630 access has negative size: {size}')
    for start,length in self.ranges:
      if start <= address and address+size <= start+length: return start,length
    raise ValueError(f'A630 access outside mapped memory: {address:#x} + {size}')
  def check(self, address:int, size:int):
    self.region(address, size)
  def child(self, ranges:tuple[tuple[int, int], ...]) -> 'Memory':
    return Memory(self.ranges+ranges, self.transactional, snapshots=self.snapshots, dirty=self.dirty, direct=self.direct+ranges)
  def _direct(self, address:int, size:int) -> bool:
    return any(start <= address and address+size <= start+length for start,length in self.direct)
  def _snapshot(self, address:int, size:int) -> tuple[bytearray, int]:
    start,length = self.region(address, size)
    key = (start,length)
    if key not in self.snapshots: self.snapshots[key] = bytearray(read_host(start, length))
    return self.snapshots[key], address-start
  def read(self, address:int, size:int) -> int:
    return int.from_bytes(self.read_bytes(address, size), 'little')
  def read_bytes(self, address:int, size:int) -> bytes:
    self.check(address, size)
    if self.transactional and not self._direct(address, size):
      snapshot,offset = self._snapshot(address, size)
      return bytes(snapshot[offset:offset+size])
    return read_host(address, size)
  def write(self, address:int, size:int, value:int):
    self.check(address, size)
    data = (value & ((1 << (size*8))-1)).to_bytes(size, 'little')
    if self.transactional and not self._direct(address, size):
      snapshot, offset = self._snapshot(address, size)
      snapshot[offset:offset+size] = data
      key = self.region(address, size)
      if key not in self.dirty: self.dirty[key] = bytearray(key[1])
      self.dirty[key][offset:offset+size] = b'\x01'*size
    else: write_host(address, data)
  def commit(self):
    if not self.transactional: raise ValueError('Cannot commit non-transactional A630 memory')
    spans:list[tuple[int, bytes]] = []
    for key,dirty in self.dirty.items():
      cursor = 0
      while (begin:=dirty.find(b'\x01', cursor)) != -1:
        end = dirty.find(b'\x00', begin)
        if end == -1: end = len(dirty)
        spans.append((key[0]+begin, bytes(self.snapshots[key][begin:end])))
        cursor = end
    for start,data in spans: validate_host_mapping(start, len(data), write=True)
    for start,data in spans: write_host(start, data)
    self.snapshots.clear()
    self.dirty.clear()

@dataclasses.dataclass(frozen=True)
class Image:
  pointer:int
  width:int
  height:int
  pitch:int
  half:bool
  swizzle:tuple[int, ...] = (0, 1, 2, 3)
  def load(self, memory:Memory, x:int, y:int) -> list[float]:
    values = [0.0]*4
    if 0 <= x < self.width and 0 <= y < self.height:
      size = 2 if self.half else 4
      offset = self.pointer+y*self.pitch+x*4*size
      values = [(bitsf16 if self.half else bitsf32)(memory.read(offset+i*size, size)) for i in range(4)]
    components = values+[0.0, 1.0]
    return [components[index] for index in self.swizzle]
  def store(self, memory:Memory, x:int, y:int, values:list[float]):
    if not (0 <= x < self.width and 0 <= y < self.height): return
    size = 2 if self.half else 4
    offset = self.pointer+y*self.pitch+x*4*size
    for i,value in enumerate(values): memory.write(offset+i*size, size, (f16bits if self.half else f32bits)(value))

# Mesa's IR3 immediate floating-point table (src/freedreno/isa/ir3-common.xml).
FLOAT_IMMEDIATES = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0)
SHARED_REGISTER_BASE, SHARED_REGISTER_END = 48*4, 56*4

class Thread:
  def __init__(self, constants:tuple[int, ...], constant_demotion:bool=False, shared_registers:Mapping[int, int]|None=None):
    self.constants, self.regs, self.half_regs = constants, [0]*256, [0]*256
    self.constant_demotion = constant_demotion
    # run_workgroup owns MappingProxyType snapshots; direct callers pass ordinary mappings that are detached and normalized here.
    self.shared_registers = (shared_registers if isinstance(shared_registers, MappingProxyType)
                             else MappingProxyType({index:value & 0xffffffff for index,value in (shared_registers or {}).items()}))
    if any(not SHARED_REGISTER_BASE <= index < SHARED_REGISTER_END for index in self.shared_registers):
      raise ValueError('A630 shared preload targets a non-shared register')
  @staticmethod
  def _check_register(index:int):
    if not 0 <= index < 256: raise ValueError(f'Invalid A630 register index {index}')
  def read_register(self, index:int, half:bool=False) -> int:
    self._check_register(index)
    if SHARED_REGISTER_BASE <= index < SHARED_REGISTER_END:
      if half: raise ValueError(f'Unsupported A630 half shared register hr{index//4}.{"xyzw"[index%4]}')
      if index not in self.shared_registers: raise ValueError(f'Unseeded A630 shared register r{index//4}.{"xyzw"[index%4]}')
      return self.shared_registers[index]
    return (self.half_regs if half else self.regs)[index]
  def source(self, operand:Operand, repeat:int=0, floating:bool=False) -> int:
    if operand.bank == 'immediate': return operand.index
    increment = repeat if operand.repeat else 0
    index = operand.index+increment
    if operand.relative: index += signed(self.half_regs[244], 16)
    if index < 0: raise ValueError('Negative A630 register index')
    if operand.bank == 'constant':
      if not operand.half: return self.constants[index]
      if not self.constant_demotion: return (self.constants[index//2] >> (16*(index%2))) & 0xffff
      return f16bits(bitsf32(self.constants[index])) if floating else self.constants[index] & 0xffff
    return self.read_register(index, operand.half)
  def float_bits(self, operand:Operand, repeat:int=0) -> int:
    bits = self.source(operand, repeat, floating=True)
    sign = 0x8000 if operand.half else 0x80000000
    if operand.absneg & 2: bits &= sign-1
    if operand.absneg & 1: bits ^= sign
    return bits
  def float_source(self, operand:Operand, repeat:int=0) -> float:
    return (bitsf16 if operand.half else bitsf32)(self.float_bits(operand, repeat))
  def signed_source(self, operand:Operand, repeat:int=0) -> int:
    value = signed(self.source(operand, repeat), 16 if operand.half else 32)
    if operand.absneg & 2: value = abs(value)
    return -value if operand.absneg & 1 else value
  def bit_source(self, operand:Operand, repeat:int=0) -> int:
    value = self.source(operand, repeat)
    return (value ^ (0xffff if operand.half else 0xffffffff)) if operand.absneg & 1 else value
  def destination(self, instruction:Instruction, value:int, repeat:int=0, half:bool|None=None):
    if half is None: half = bool(instruction.field('DST_HALF'))
    relative = signed(self.half_regs[244], 16) if instruction.field('DST_REL') else 0
    self.write(instruction.field('DST')+relative+repeat, value, half)
  def write(self, index:int, value:int, half:bool=False):
    self._check_register(index)
    if SHARED_REGISTER_BASE <= index < SHARED_REGISTER_END:
      raise ValueError(f'Unsupported A630 shared register write to {"hr" if half else "r"}{index//4}.{"xyzw"[index%4]}')
    registers, mask = (self.half_regs, 0xffff) if half else (self.regs, 0xffffffff)
    registers[index] = value & mask
  def float_destination(self, instruction:Instruction, value:float, repeat:int=0):
    if instruction.field('SAT'): value = min(1.0, max(0.0, value))
    self.destination(instruction, (f16bits if instruction.field('DST_HALF') else f32bits)(value), repeat)

def run_thread(program:bytes, constants:tuple[int, ...], memory:Memory, initial_registers:dict[int, int]|None=None,
               shared:int=0, private:int=0, textures:tuple[Image, ...]=(), images:tuple[Image, ...]=(), start:int=0,
               constant_demotion:bool=False, initial_half_registers:dict[int, int]|None=None,
               shared_registers:Mapping[int, int]|None=None, shared_size:int=0, private_size:int=0) -> Generator[int, None, Thread]:
  thread = Thread(constants, constant_demotion, shared_registers)
  for index,value in (initial_registers or {}).items(): thread.write(index, value)
  for index,value in (initial_half_registers or {}).items(): thread.write(index, value, True)
  instructions, pc = decode(program), start
  returns:list[int] = []
  predication:bool|None = None
  predicate = False
  max_steps, max_call_depth = int(os.getenv('MOCK_QCOM_MAX_STEPS', '20000000')), int(os.getenv('MOCK_QCOM_MAX_CALL_DEPTH', '64'))
  if max_steps <= 0 or max_call_depth <= 0: raise ValueError('A630 execution budgets must be positive')
  steps = 0
  while 0 <= pc < len(instructions):
    steps += 1
    if steps > max_steps: raise RuntimeError(f'A630 instruction budget exhausted ({max_steps} steps)')
    current_pc, instruction = pc, instructions[pc]
    pc += 1
    name = instruction.name
    if instruction.field('JP'): predicate = bool(thread.regs[248])
    if name in ('predt', 'predf', 'prede'):
      predication = None if name == 'prede' else name == 'predt'
      predicate = bool(thread.regs[248])
      continue
    if predication is not None and predicate != predication: continue
    if name == 'end': return thread
    if instruction.field('EI') and name != 'add.u': raise ValueError(f'Unsupported A630 EI modifier on {name}')
    if name == 'call':
      if len(returns) >= max_call_depth: raise RuntimeError(f'A630 call stack budget exhausted ({max_call_depth} calls)')
      returns.append(pc)
      pc = current_pc+signed(instruction.field('IMMED'))
      continue
    if name == 'ret':
      if not returns: raise ValueError('A630 return without a call')
      pc = returns.pop()
      continue
    if name in ('nop', 'fence'): continue
    if name == 'bar':
      yield current_pc
      continue
    if name in ('br', 'brao', 'braa', 'jump'):
      condition = bool(thread.regs[248+instruction.field('COMP1')]) != bool(instruction.field('INV1'))
      if name in ('brao', 'braa'):
        second = bool(thread.regs[248+instruction.field('COMP2')]) != bool(instruction.field('INV2'))
        condition = condition or second if name == 'brao' else condition and second
      if name == 'jump' or condition: pc = current_pc+signed(instruction.field('IMMED'))
      continue
    for repeat in range(instruction.field('REPEAT')+1):
      if name == 'mov':
        value = convert(thread.source(instruction.operand('SRC'), repeat, floating=instruction.field('SRC_TYPE') == 0), instruction.field('SRC_TYPE'),
                        instruction.field('DST_TYPE'), instruction.field('ROUND'))
        thread.destination(instruction, value, repeat)
      elif name in ('swz', 'gat', 'sct'):
        src = ([instruction.field('SRC0')+i for i in range(4)] if name == 'sct'
               else [instruction.field('SRC'+str(i)) for i in range(2 if name == 'swz' else 4)])
        dst = ([instruction.field('DST0')+i for i in range(4)] if name == 'gat'
               else [instruction.field('DST'+str(i)) for i in range(2 if name == 'swz' else 4)])
        # Swaps and overlapping gathers/scatters must read all old values before the first write.
        values = [convert(thread.read_register(index, bool(instruction.field('HALF'))), instruction.field('SRC_TYPE'),
                          instruction.field('DST_TYPE'), instruction.field('ROUND'))
                  for index in src]
        for index,value in zip(dst, values): thread.write(index, value, bool(instruction.field('DST_HALF')))
      elif name in ('add.f', 'mul.f', 'min.f', 'max.f'):
        left, right = [thread.float_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        if name == 'add.f': float_value = left+right
        elif name == 'mul.f': float_value = left*right
        elif name == 'min.f': float_value = min(left, right)
        else: float_value = max(left, right)
        thread.float_destination(instruction, float_value, repeat)
      elif name == 'absneg.f':
        operand = instruction.operand('SRC1')
        thread.float_destination(instruction, thread.float_source(operand, repeat), repeat)
      elif name in ('rcp', 'rsq', 'sqrt', 'exp2', 'log2', 'sin', 'cos', 'hrsq', 'hexp2', 'hlog2'):
        thread.float_destination(instruction, special_float(name.removeprefix('h'), thread.float_source(instruction.operand('SRC'), repeat)), repeat)
      elif name in ('floor.f', 'ceil.f', 'trunc.f', 'rndne.f'):
        thread.float_destination(instruction, special_float(name, thread.float_source(instruction.operand('SRC1'), repeat)), repeat)
      elif name == 'absneg.s': thread.destination(instruction, thread.signed_source(instruction.operand('SRC1'), repeat), repeat)
      elif name == 'not.b': thread.destination(instruction, ~thread.bit_source(instruction.operand('SRC1'), repeat), repeat)
      elif name in ('clz.b', 'clz.s'):
        operand = instruction.operand('SRC1')
        bits = thread.signed_source(operand, repeat) if name == 'clz.s' else thread.bit_source(operand, repeat)
        if bits < 0: bits = ~bits
        # Unlike a CPU CLZ, Adreno returns -1 when no qualifying bit exists (Mesa's find_msb/find_lsb lowering).
        thread.destination(instruction, (16 if operand.half else 32)-bits.bit_length() if bits else -1, repeat)
      elif name == 'sign.f':
        number = thread.float_source(instruction.operand('SRC1'), repeat)
        thread.float_destination(instruction, number if number == 0 else float((number > 0)-(number < 0)), repeat)
      elif name in ('min.s', 'max.s'):
        lhs, rhs = [thread.signed_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        thread.destination(instruction, min(lhs, rhs) if name == 'min.s' else max(lhs, rhs), repeat)
      elif name in ('add.u', 'add.s', 'sub.u', 'sub.s', 'min.u', 'max.u', 'and.b', 'or.b', 'xor.b', 'shl.b', 'shr.b', 'ashr.b',
                    'mull.u', 'mul.u24', 'mul.s24', 'getbit.b'):
        reader = thread.signed_source if name.endswith('.s') else thread.bit_source if name.endswith('.b') else thread.source
        left, right = [reader(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2')]
        if instruction.field('EI'):
          # OpenCL hadd emits full-width ADD.U(EI): retain the carry until the mathematical sum is halved.
          if name != 'add.u' or instruction.field('DST_HALF') or instruction.field('SAT'):
            raise ValueError(f'Unsupported A630 EI modifier on {name}')
          value = (left+right) >> 1
        else:
          if name in ('add.u', 'add.s'): value = left+right
          elif name in ('sub.u', 'sub.s'): value = left-right
          elif name == 'min.u': value = min(left, right)
          elif name == 'max.u': value = max(left, right)
          elif name == 'and.b': value = left & right
          elif name == 'or.b': value = left | right
          elif name == 'xor.b': value = left ^ right
          elif name == 'shl.b': value = left << (right & 31)
          elif name == 'shr.b': value = left >> (right & 31)
          elif name == 'ashr.b': value = signed(left, 16 if instruction.operand('SRC1').half else 32) >> (right & 31)
          elif name == 'getbit.b': value = (left >> (right & 31)) & 1
          elif name == 'mull.u': value = (left & 0xffff)*(right & 0xffff)
          elif name == 'mul.u24': value = (left & 0xffffff)*(right & 0xffffff)
          else: value = (signed(left, 16 if instruction.operand('SRC1').half else 24) *
                         signed(right, 16 if instruction.operand('SRC2').half else 24))
          if instruction.field('SAT'):
            if name not in ('add.u', 'add.s', 'sub.u', 'sub.s'): raise ValueError(f'Unsupported A630 integer saturation on {name}')
            width = 16 if instruction.field('DST_HALF') else 32
            low, high = (-(1 << (width-1)), (1 << (width-1))-1) if name.endswith('.s') else (0, (1 << width)-1)
            value = min(high, max(low, value))
        thread.destination(instruction, value, repeat)
      elif name in ('madsh.m16', 'madsh.u16'):
        a, b, addend = [thread.source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, (a & 0xffff)*(b & 0xffff0000)+addend, repeat)
      elif name in ('mad.u16', 'mad.s16', 'mad.u24', 'mad.s24'):
        width = int(name[-2:])
        operands = [instruction.operand(key) for key in ('SRC1', 'SRC2', 'SRC3')]
        numbers = [thread.signed_source(operand, repeat) if '.s' in name else thread.source(operand, repeat) for operand in operands]
        a, b = [signed(value, width) if '.s' in name else value & ((1 << width)-1) for value in numbers[:2]]
        thread.destination(instruction, a*b+numbers[2], repeat)
      elif name in ('sad.s16', 'sad.s32'):
        # Mesa lowers iadd3 to SAD: these instructions add three signed operands.
        thread.destination(instruction, sum(thread.signed_source(instruction.operand(key), repeat) for key in ('SRC1','SRC2','SRC3')), repeat)
      elif name in ('mad.f32', 'mad.f16'):
        fa, fb, fc = [thread.float_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        # A6xx MAD is unfused: Mesa's ir3_compiler_nir.c permits lowering it to MUL followed by ADD.
        product = bitsf16(f16bits(fa*fb)) if name == 'mad.f16' else bitsf32(f32bits(fa*fb))
        thread.float_destination(instruction, product+fc, repeat)
      elif name in ('sel.b32', 'sel.b16'):
        yes, selector, no = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, yes if selector else no, repeat)
      elif name in ('sel.f32', 'sel.f16'):
        # The math library uses an ordered floating predicate, including NaN as the false case.
        operand = instruction.operand('SRC1' if thread.float_source(instruction.operand('SRC2'), repeat) >= 0 else 'SRC3')
        thread.float_destination(instruction, thread.float_source(operand, repeat), repeat)
      elif name in ('sel.s32', 'sel.s16'):
        operand = instruction.operand('SRC1' if thread.signed_source(instruction.operand('SRC2'), repeat) >= 0 else 'SRC3')
        thread.destination(instruction, thread.signed_source(operand, repeat), repeat)
      elif name in ('cmps.u', 'cmps.s', 'cmps.f', 'cmpv.u', 'cmpv.s', 'cmpv.f'):
        operands = [instruction.operand(key) for key in ('SRC1', 'SRC2')]
        if name.endswith('.f'): left, right = [thread.float_source(operand, repeat) for operand in operands]
        elif name.endswith('.s'): left, right = [thread.signed_source(operand, repeat) for operand in operands]
        else: left, right = [thread.source(operand, repeat) for operand in operands]
        result = (left < right, left <= right, left > right, left >= right, left == right, left != right)[instruction.field('COND')]
        # Bit 42 inverts comparisons, although this Mesa decoder labels it SAT like the arithmetic instructions.
        if instruction.field('SAT'): result = not result
        thread.destination(instruction, -int(result) if name.startswith('cmpv') else int(result), repeat)
      elif name in ('shrg', 'shlg', 'shrm', 'shlm'):
        shift, value, extra = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        value = (value >> (shift & 31)) if name in ('shrg', 'shrm') else (value << (shift & 31))
        value = value & extra if name in ('shrm', 'shlm') else value | extra
        thread.destination(instruction, value, repeat)
      elif name == 'andg':
        numbers = [thread.bit_source(instruction.operand(key), repeat) for key in ('SRC1', 'SRC2', 'SRC3')]
        thread.destination(instruction, (numbers[1] & numbers[0]) | numbers[2], repeat)
      elif name == 'isam':
        if any(instruction.field(flag) for flag in ('3D','A','O','P','S2EN','S2EN_BINDLESS')):
          raise ValueError('Unsupported A630 texture addressing mode')
        coordinate, coordinate_half = instruction.field('SRC1'), bool(instruction.field('HALF'))
        coordinate_bits = 16 if coordinate_half else 32
        texels = textures[instruction.field('TEX')].load(memory, signed(thread.read_register(coordinate, coordinate_half), coordinate_bits),
                                                         signed(thread.read_register(coordinate+1, coordinate_half), coordinate_bits))
        for component,texel in enumerate(texels):
          if instruction.field('WRMASK') & (1 << component):
            # Cat5 write masks preserve component positions. A green-only load writes DST+1, not a packed value at DST.
            thread.float_destination(instruction, texel, component)
      elif name in ('stib.b', 'ldib.b'):
        if instruction.field('MODE') or instruction.field('D') != 2: raise ValueError('Unsupported A630 image addressing mode')
        resource = images[instruction.field('SSBO')]
        coordinate, half = instruction.field('SRC2'), bool(instruction.field('TYPE_HALF'))
        x, y = signed(thread.read_register(coordinate)), signed(thread.read_register(coordinate+1))
        count, register = instruction.field('TYPE_SIZE'), instruction.field('SRC1')
        if name == 'stib.b':
          texels = [(bitsf16 if half else bitsf32)(thread.read_register(register+i, half)) for i in range(count)]
          resource.store(memory, x, y, texels)
        else:
          for i,texel in enumerate(resource.load(memory, x, y)[:count]):
            thread.write(register+i, (f16bits if half else f32bits)(texel), half)
      elif name in ('ldg', 'stg', 'ldg.a', 'stg.a', 'ldl', 'stl', 'ldp', 'stp'):
        kind = instruction.field('TYPE')
        width = (2, 4, 2, 4, 2, 4, 1, 1)[kind]
        load = name.startswith('ld')
        offset = instruction.field('OFF') if name.endswith('.a') else signed(instruction.field('OFF'), 13)
        if name.startswith(('ldg', 'stg')):
          pointer_reg = instruction.field('SRC1')
          address = thread.read_register(pointer_reg) | (thread.read_register(pointer_reg+1) << 32)
          source = instruction.field('SRC3')
          if name.endswith('.a'): offset = ((thread.read_register(instruction.field('SRC2')) << instruction.field('SRC2_SHIFT'))+offset)*width
          address += offset
        else:
          space,base,limit = ('local', shared, shared_size) if name.endswith('l') else ('private', private, private_size)
          space_offset = thread.read_register(instruction.field('SRC' if load else 'DST'))+offset
          source = instruction.field('SRC')
          access_size = instruction.field('SIZE')*width
          if space_offset < 0 or space_offset+access_size > limit:
            raise ValueError(f'A630 {space} access outside allocated memory: {space_offset:#x} + {access_size}')
          address = base+space_offset
        memory.check(address, instruction.field('SIZE')*width)
        for component in range(instruction.field('SIZE')):
          if load:
            thread.destination(instruction, memory.read(address+component*width, width), component, half=kind in (0,2,4,6,7))
          else:
            memory.write(address+component*width, width, thread.read_register(source+component, kind in (0,2,4,6,7)))
      else: raise ValueError(f'A630 instruction is not implemented: {name} ({instruction.word:#x})')
  raise ValueError('A630 instruction stream ended without an end instruction')

def run_scalar(program:bytes, constants:tuple[int, ...], memory:Memory, initial_registers:dict[int, int]|None=None) -> Thread:
  worker = run_thread(program, constants, memory, initial_registers)
  try: next(worker)
  except StopIteration as completed: return completed.value
  finally: worker.close()
  raise ValueError('A barrier requires workgroup execution')

def run_workgroup(program:bytes, constants:tuple[int, ...], memory:Memory, registers:list[dict[int, int]], shared_size:int, private_size:int,
                  textures:tuple[Image, ...]=(), images:tuple[Image, ...]=(), start:int=0, constant_demotion:bool=False,
                  shared_registers:Mapping[int, int]|None=None):
  shared_registers = MappingProxyType({index:value & 0xffffffff for index,value in (shared_registers or {}).items()})
  if any(not SHARED_REGISTER_BASE <= index < SHARED_REGISTER_END for index in shared_registers):
    raise ValueError('A630 shared preload targets a non-shared register')
  shared = ctypes.create_string_buffer(shared_size)
  private = [ctypes.create_string_buffer(private_size) for _ in registers]
  shared_address = ctypes.addressof(shared)
  workers = [run_thread(program, constants,
                        memory.child(((shared_address, shared_size), (ctypes.addressof(scratch), private_size))),
                        initial, shared_address, ctypes.addressof(scratch), textures, images, start, constant_demotion,
                        shared_registers=shared_registers, shared_size=shared_size, private_size=private_size)
             for initial,scratch in zip(registers, private)]
  try:
    while workers:
      waiting = []
      for worker in workers:
        try: waiting.append((worker, next(worker)))
        except StopIteration: pass
      if waiting and len(waiting) != len(workers): raise RuntimeError('A630 work-item exited before the workgroup barrier')
      if len({pc for _,pc in waiting}) > 1: raise RuntimeError('A630 workgroup reached different barrier instructions')
      workers = [worker for worker,_ in waiting]
  finally:
    for worker in workers: worker.close()
