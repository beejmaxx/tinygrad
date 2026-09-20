import collections, ctypes, functools, mmap, os, threading
from dataclasses import dataclass, field
from typing import cast
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import GPUState, Packet, QCOMGPU
from test.mockgpu.qcom.host import read_struct, validate_host_mapping, write_host

SUPPORTED_IOCTLS = ('DEVICE_GETPROPERTY', 'DEVICE_WAITTIMESTAMP_CTXTID', 'DRAWCTXT_CREATE', 'DRAWCTXT_DESTROY', 'MAP_USER_MEM',
                    'CMDSTREAM_READTIMESTAMP_CTXTID', 'SHAREDMEM_FREE', 'SETPROPERTY', 'GPUOBJ_ALLOC', 'GPUOBJ_FREE', 'GPU_COMMAND')
def ioctl_code(request:functools.partial) -> int:
  direction, base, number, record_type = request.args
  return direction << 30 | (ctypes.sizeof(record_type) if record_type is not None else 0) << 16 | base << 8 | number
IOCTLS = {ioctl_code(request): ('IOCTL_KGSL_'+suffix, request.args[3]) for suffix in SUPPORTED_IOCTLS
          if isinstance(request:=getattr(kgsl, 'IOCTL_KGSL_'+suffix), functools.partial)}

@dataclass
class Allocation:
  size:int
  pointer:int = 0
  owner_fd:int|None = None

@dataclass
class ExternalMapping:
  owners:dict[int|None, tuple[int, int]] = field(default_factory=dict)
  def add(self, fd:int|None, size:int):
    references,extent = self.owners.get(fd, (0, 0))
    self.owners[fd] = references+1,max(extent, size)
  def remove(self, fd:int|None):
    references,extent = self.owners[fd]
    if references == 1: del self.owners[fd]
    else: self.owners[fd] = references-1,extent
  @property
  def size(self) -> int: return max(size for _,size in self.owners.values())
  @property
  def references(self) -> int: return sum(references for references,_ in self.owners.values())

@dataclass
class Context:
  owner_fd:int|None = None
  queued:int = 0
  consumed:int = 0
  retired:int = 0
  pending:collections.deque[tuple[tuple[Packet, ...], int, int]] = field(default_factory=collections.deque)
  state:GPUState = field(default_factory=GPUState)
  fault:Exception|None = None

class QCOMFileDesc(VirtFileDesc):
  def __init__(self, fd:int, driver:'QCOMDriver'):
    super().__init__(fd)
    self.driver = driver
  def ioctl(self, fd, request, argp):
    self.raise_if_failed()
    return self.driver.ioctl(request, argp, fd)
  def mmap(self, start, size, prot, flags, fd, offset):
    return self.driver.mmap(fd, start, size, prot, flags, offset)
  def close(self, fd):
    try: self.driver.close(fd)
    finally: os.close(fd)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files = [VirtFile('/dev/kgsl-3d0', functools.partial(QCOMFileDesc, driver=self))]
    self.allocations:dict[int, Allocation] = {}
    self.external:dict[int, ExternalMapping] = {}
    self.next_allocation = self.next_context = 1
    self.contexts:dict[int, Context] = {}
    self.lock, self.draining = threading.RLock(), False
    self.gpu = QCOMGPU(self.ranges)
  def ranges(self, fd:int|None=None) -> tuple[tuple[int, int], ...]:
    with self.lock:
      allocations = tuple((a.pointer, a.size) for a in self.allocations.values() if a.pointer and (fd is None or a.owner_fd == fd))
      external = tuple((pointer, mapping.size if fd is None else mapping.owners[fd][1])
                       for pointer,mapping in self.external.items() if fd is None or fd in mapping.owners)
      return allocations+external
  def open(self, name, flags, mode, virtfile):
    # An OS-allocated placeholder descriptor cannot collide with another mock driver's descriptor range.
    return virtfile.fdcls(os.open(os.devnull, os.O_RDWR))
  def context(self, context_id:int, fd:int|None) -> Context:
    if context_id not in self.contexts or self.contexts[context_id].owner_fd != fd: raise ValueError(f'Unknown QCOM context {context_id}')
    return self.contexts[context_id]
  @staticmethod
  def raise_fault(context:Context):
    if context.fault is not None: raise context.fault
  def mmap(self, fd:int|None, start:int, size:int, prot:int, flags:int, offset:int) -> int:
    with self.lock:
      if offset % 0x1000 or (allocation:=self.allocations.get(offset//0x1000)) is None or allocation.owner_fd != fd:
        raise ValueError(f'Unknown QCOM allocation mapping offset {offset:#x}')
      if start or allocation.pointer or size != allocation.size: raise ValueError('Invalid QCOM allocation mapping')
      pointer = libc.mmap(0, size, prot, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
      if pointer == ctypes.c_void_p(-1).value: raise OSError('QCOM mock mmap failed')
      allocation.pointer = pointer
      return pointer
  def close(self, fd:int):
    with self.lock:
      self.contexts = {key:value for key,value in self.contexts.items() if value.owner_fd != fd}
      for key,allocation in list(self.allocations.items()):
        if allocation.owner_fd != fd: continue
        if allocation.pointer: libc.munmap(allocation.pointer, allocation.size)
        del self.allocations[key]
      for pointer,mapping in list(self.external.items()):
        mapping.owners.pop(fd, None)
        if not mapping.owners: del self.external[pointer]
  def drain(self):
    with self.lock:
      if self.draining: return
      self.draining = True
      try:
        progress = True
        while progress:
          progress = False
          for context in self.contexts.values():
            if context.fault is not None or not context.pending: continue
            packets,timestamp,cursor = context.pending[0]
            self.gpu.load_state(context.state)
            try:
              complete,next_cursor = self.gpu.execute_packets(packets, cursor, self.ranges(context.owner_fd))
            except Exception as error:
              context.consumed, context.fault = timestamp, error
              context.pending.clear()
              continue
            finally: context.state = self.gpu.save_state()
            context.consumed = timestamp
            if complete:
              context.pending.popleft()
              context.retired, progress = timestamp, True
            else:
              context.pending[0] = packets,timestamp,next_cursor
              progress |= next_cursor != cursor
      finally: self.draining = False
  def ioctl(self, request:int, pointer:int, fd:int|None=None) -> int:
    with self.lock: return self._ioctl(request, pointer, fd)
  def _ioctl(self, request:int, pointer:int, fd:int|None=None) -> int:
    if pointer == 0: raise ValueError('QCOM ioctl record pointer is null')
    if (entry:=IOCTLS.get(request)) is None: raise ValueError(f'Unsupported QCOM ioctl {request:#x}')
    name, record_type = entry
    record = read_struct(pointer, record_type, writable=True)
    if name == 'IOCTL_KGSL_DRAWCTXT_CREATE':
      record.drawctxt_id = self.next_context
      self.contexts[self.next_context] = Context(fd)
      self.next_context += 1
    elif name == 'IOCTL_KGSL_DRAWCTXT_DESTROY':
      if self.context(record.drawctxt_id, fd).pending: raise ValueError('QCOM context has pending commands')
      del self.contexts[record.drawctxt_id]
    elif name == 'IOCTL_KGSL_DEVICE_GETPROPERTY':
      if record.type != kgsl.KGSL_PROP_DEVICE_INFO: raise ValueError(f'Unsupported QCOM property {record.type}')
      if not record.value or record.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError('Invalid QCOM device info buffer')
      info = kgsl.struct_kgsl_devinfo()
      info.device_id, info.chip_id, info.gpu_id, info.gmem_sizebytes, info.mmu_enabled = 1, 0x06030001, 630, 1 << 20, 1
      write_host(record.value, bytes(info))
    elif name == 'IOCTL_KGSL_SETPROPERTY':
      if record.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise ValueError(f'Unsupported QCOM property {record.type}')
      if not record.value or record.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_device_constraint):
        raise ValueError('Invalid QCOM power constraint buffer')
      constraint = read_struct(record.value, kgsl.struct_kgsl_device_constraint)
      self.context(constraint.context_id, fd)
      if constraint.type != kgsl.KGSL_CONSTRAINT_PWRLEVEL: raise ValueError(f'Unsupported QCOM power constraint {constraint.type}')
      if not constraint.data or constraint.size < ctypes.sizeof(kgsl.struct_kgsl_device_constraint_pwrlevel):
        raise ValueError('Invalid QCOM power level buffer')
      level = read_struct(cast(int, constraint.data), kgsl.struct_kgsl_device_constraint_pwrlevel)
      if level.level != kgsl.KGSL_CONSTRAINT_PWR_MAX: raise ValueError(f'Unsupported QCOM power level {level.level}')
    elif name == 'IOCTL_KGSL_GPUOBJ_ALLOC':
      if record.size == 0: raise ValueError('QCOM allocation size must be nonzero')
      record.id = self.next_allocation
      record.mmapsize = record.size
      self.allocations[record.id] = Allocation(record.size, owner_fd=fd)
      self.next_allocation += 1
    elif name == 'IOCTL_KGSL_GPUOBJ_FREE':
      if record.id not in self.allocations or self.allocations[record.id].owner_fd != fd:
        raise ValueError(f'Unknown QCOM allocation {record.id}')
      if any(context.owner_fd == fd and context.pending for context in self.contexts.values()):
        raise ValueError('Cannot free QCOM memory with pending commands')
      del self.allocations[record.id]
    elif name == 'IOCTL_KGSL_MAP_USER_MEM':
      if record.memtype != kgsl.KGSL_USER_MEM_TYPE_ADDR: raise ValueError(f'Unsupported QCOM user memory type {record.memtype}')
      if record.hostptr == 0 or record.len == 0 or record.offset: raise ValueError('Invalid QCOM user memory mapping')
      validate_host_mapping(record.hostptr, record.len, write=True)
      record.gpuaddr = record.hostptr
      allocation_ranges = ((allocation.pointer, allocation.size) for allocation in self.allocations.values() if allocation.pointer)
      external_ranges = ((start, mapping.size) for start,mapping in self.external.items() if start != record.hostptr)
      if any(record.hostptr < start+length and start < record.hostptr+record.len for start,length in (*allocation_ranges, *external_ranges)):
        raise ValueError('Overlapping QCOM user memory mappings are unsupported')
      if (mapping:=self.external.get(record.hostptr)) is not None:
        mapping.add(fd, record.len)
      else:
        mapping = self.external[record.hostptr] = ExternalMapping()
        mapping.add(fd, record.len)
    elif name == 'IOCTL_KGSL_SHAREDMEM_FREE':
      if (mapping:=self.external.get(record.gpuaddr)) is None: raise ValueError(f'Unknown QCOM user memory mapping {record.gpuaddr:#x}')
      if any(context.owner_fd == fd and context.pending for context in self.contexts.values()):
        raise ValueError('Cannot unmap QCOM memory with pending commands')
      if fd not in mapping.owners:
        raise ValueError(f'Unknown QCOM user memory mapping {record.gpuaddr:#x}')
      # The ABI identifies only the shared GPU address, not an individual same-address registration. Keep the largest validated
      # extent visible to this descriptor until its final reference is released; guessing FIFO or LIFO can invalidate a live alias.
      mapping.remove(fd)
      if not mapping.owners: del self.external[record.gpuaddr]
    elif name == 'IOCTL_KGSL_GPU_COMMAND':
      context = self.context(record.context_id, fd)
      self.raise_fault(context)
      if record.cmdsize != ctypes.sizeof(kgsl.struct_kgsl_command_object) or record.numcmds == 0 or record.cmdlist == 0:
        raise ValueError('Invalid QCOM command list')
      if record.numobjs or record.numsyncs: raise ValueError('QCOM object and sync lists are not implemented')
      packets:list[Packet] = []
      for i in range(record.numcmds):
        command = read_struct(record.cmdlist+i*record.cmdsize, kgsl.struct_kgsl_command_object)
        if command.flags != kgsl.KGSL_CMDLIST_IB or command.size == 0 or command.size % 4:
          raise ValueError('Invalid QCOM indirect command buffer')
        packets.extend(self.gpu.capture(command.gpuaddr+command.offset, command.size, self.ranges(fd)))
      context.queued += 1
      record.timestamp = context.queued
      context.pending.append((tuple(packets), record.timestamp, 0))
      self.drain()
      self.raise_fault(context)
    elif name == 'IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID':
      context = self.context(record.context_id, fd)
      if record.type not in (kgsl.KGSL_TIMESTAMP_CONSUMED, kgsl.KGSL_TIMESTAMP_RETIRED, kgsl.KGSL_TIMESTAMP_QUEUED):
        raise ValueError(f'Unsupported QCOM timestamp type {record.type}')
      self.drain()
      self.raise_fault(context)
      record.timestamp = {kgsl.KGSL_TIMESTAMP_QUEUED:context.queued, kgsl.KGSL_TIMESTAMP_CONSUMED:context.consumed,
                          kgsl.KGSL_TIMESTAMP_RETIRED:context.retired}[record.type]
    elif name == 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID':
      context = self.context(record.context_id, fd)
      self.drain()
      self.raise_fault(context)
      if record.timestamp > context.retired: raise RuntimeError('QCOM timestamp has not completed')
    else: raise ValueError(f'Unsupported QCOM request {name}')
    write_host(pointer, bytes(record))
    return 0
