"""Checked access to host memory registered with the QCOM mock."""
import ctypes, os, sys
from tinygrad.runtime.autogen import libc

MAX_HOST_SPAN = 1 << 30

class IOVec(ctypes.Structure):
  _fields_ = [('base', ctypes.c_void_p), ('size', ctypes.c_size_t)]

def _check_span(address:int, size:int):
  if address <= 0 or not 0 <= size <= MAX_HOST_SPAN or address+size > 1 << 64:
    raise ValueError(f'Invalid QCOM host memory span {address:#x} + {size}')

def validate_host_mapping(address:int, size:int, write:bool=False):
  _check_span(address, size)
  if size == 0: return
  end, cursor = address+size, address
  if sys.platform == 'linux':
    with open('/proc/self/maps') as maps:
      for line in maps:
        extent, permissions = line.split()[:2]
        start, stop = (int(part, 16) for part in extent.split('-'))
        if start <= cursor < stop and permissions[0] == 'r' and (not write or permissions[1] == 'w'):
          cursor = min(stop, end)
        if cursor == end: return
  elif sys.platform == 'darwin':
    task = ctypes.c_uint.in_dll(libc.dll, 'mach_task_self_').value
    while cursor < end:
      start, length, count = ctypes.c_uint64(cursor), ctypes.c_uint64(), ctypes.c_uint(9)
      info, object_name = (ctypes.c_int*9)(), ctypes.c_uint()
      result = libc.dll.mach_vm_region(ctypes.c_uint(task), ctypes.byref(start), ctypes.byref(length), ctypes.c_int(9),
                                       info, ctypes.byref(count), ctypes.byref(object_name))
      if object_name.value: libc.dll.mach_port_deallocate(ctypes.c_uint(task), object_name)
      required = 3 if write else 1
      if result or start.value > cursor or not length.value or info[0] & required != required: break
      cursor = min(start.value+length.value, end)
    if cursor == end: return
  else: raise NotImplementedError('QCOM MockGPU host memory requires Linux or macOS')
  raise ValueError(f'QCOM host memory is not live with requested permissions: {address:#x} + {size}')

def _transfer(address:int, data:bytearray, write:bool):
  validate_host_mapping(address, len(data), write)
  if not data: return
  local = (ctypes.c_ubyte*len(data)).from_buffer(data)
  source, destination = (ctypes.addressof(local), address) if write else (address, ctypes.addressof(local))
  if sys.platform == 'darwin':
    task, copied = ctypes.c_uint.in_dll(libc.dll, 'mach_task_self_').value, ctypes.c_uint64()
    result = libc.dll.mach_vm_read_overwrite(ctypes.c_uint(task), ctypes.c_uint64(source), ctypes.c_uint64(len(data)),
                                             ctypes.c_uint64(destination), ctypes.byref(copied))
    complete = result == 0 and copied.value == len(data)
  else:
    source_vec, destination_vec = IOVec(source, len(data)), IOVec(destination, len(data))
    libc.dll.process_vm_readv.restype = ctypes.c_ssize_t
    copied = libc.dll.process_vm_readv(ctypes.c_int(os.getpid()), ctypes.byref(destination_vec), ctypes.c_ulong(1),
                                      ctypes.byref(source_vec), ctypes.c_ulong(1), ctypes.c_ulong(0))
    complete = copied == len(data)
  if not complete: raise ValueError(f'QCOM host memory {"write" if write else "read"} failed at {address:#x}')

def read_host(address:int, size:int) -> bytes:
  _check_span(address, size)
  data = bytearray(size)
  _transfer(address, data, False)
  return bytes(data)

def write_host(address:int, data:bytes|bytearray):
  _transfer(address, bytearray(data), True)

def read_struct(address:int, typ, writable:bool=False):
  alignment = max(ctypes.alignment(field[1]) for field in typ._real_fields_)
  if address % alignment: raise ValueError(f'Misaligned QCOM structure at {address:#x}')
  validate_host_mapping(address, ctypes.sizeof(typ), writable)
  return typ.from_buffer_copy(read_host(address, ctypes.sizeof(typ)))
