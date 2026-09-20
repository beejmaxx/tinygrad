import ctypes, mmap, threading, time, unittest
from unittest.mock import patch
from tinygrad.runtime.autogen import kgsl, mesa
from tinygrad.runtime.ops_qcom import pkt4_hdr, pkt7_hdr
from test.mockgpu.qcom import emu, host
from test.mockgpu.qcom.emu import Memory
from test.mockgpu.qcom.qcomdriver import QCOMDriver, QCOMFileDesc, ioctl_code

def ioctl(driver:QCOMDriver, request, record, fd=None):
  return driver.ioctl(ioctl_code(request), ctypes.addressof(record), fd)

class TestQCOMDriver(unittest.TestCase):
  def test_unknown_records_and_mappings_are_rejected(self):
    driver = QCOMDriver()
    with self.assertRaisesRegex(ValueError, 'record pointer is null'): driver.ioctl(ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC), 0)
    with self.assertRaisesRegex(ValueError, 'Unsupported QCOM ioctl'): driver.ioctl(0xff, 1)
    descriptor = QCOMFileDesc(-1, driver)
    with self.assertRaisesRegex(ValueError, 'mapping offset'): descriptor.mmap(0, 4096, 0, 0, -1, 1)
    with self.assertRaisesRegex(ValueError, 'mapping offset'): descriptor.mmap(0, 4096, 0, 0, -1, 0x1000)

  def test_unmapped_host_pointer_is_rejected_without_dereference(self):
    driver = QCOMDriver()
    mapping = kgsl.struct_kgsl_map_user_mem(hostptr=1, len=4096, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    with self.assertRaisesRegex(ValueError, 'host memory is not live'): ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM, mapping)
    self.assertNotIn(1, driver.external)
    with self.assertRaisesRegex(ValueError, 'host memory is not live'): Memory(((1, 4),)).read(1, 4)

  def test_transaction_snapshots_each_region_once_and_commits_only_writes(self):
    source, output = (ctypes.c_uint32*2)(11, 22), (ctypes.c_uint32*2)(0, 44)
    source_pointer, output_pointer = ctypes.addressof(source), ctypes.addressof(output)
    memory = Memory(((source_pointer, ctypes.sizeof(source)), (output_pointer, ctypes.sizeof(output))), transactional=True)
    with patch.object(emu, 'read_host', wraps=emu.read_host) as read, patch.object(emu, 'write_host', wraps=emu.write_host) as write:
      self.assertEqual((memory.read(source_pointer, 4), memory.read(source_pointer+4, 4)), (11, 22))
      self.assertEqual(read.call_count, 1)
      memory.write(output_pointer, 4, 33)
      output[1] = 77  # This byte range shares a mapping but is not part of the GPU write.
      self.assertEqual(tuple(output), (0, 77))
      memory.commit()
      self.assertEqual(tuple(output), (33, 77))
      write.assert_called_once()

  def test_transaction_reuses_dirty_bitmap(self):
    data = (ctypes.c_uint32*2)()
    pointer = ctypes.addressof(data)
    memory = Memory(((pointer, ctypes.sizeof(data)),), transactional=True)
    memory.write(pointer, 4, 1)
    allocations = []
    def allocate(size):
      allocations.append(size)
      return bytearray(size)
    with patch.object(emu, 'bytearray', allocate, create=True):
      for value in range(2, 18): memory.write(pointer, 4, value)
    self.assertEqual(allocations, [])
    self.assertEqual(len(memory.dirty), 1)

  def test_oversized_host_read_is_rejected_before_allocation(self):
    with patch.object(host, 'bytearray', side_effect=AssertionError('oversized allocation'), create=True) as allocation:
      with self.assertRaisesRegex(ValueError, 'Invalid QCOM host memory span'):
        host.read_host(1, host.MAX_HOST_SPAN+1)
    allocation.assert_not_called()

  def test_memory_ranges_are_canonical_and_nonoverlapping(self):
    data = (ctypes.c_uint8*16)()
    pointer = ctypes.addressof(data)
    self.assertEqual(Memory(((pointer, 8), (pointer, 16))).ranges, ((pointer, 16),))
    with self.assertRaisesRegex(ValueError, 'ranges overlap'): Memory(((pointer, 8), (pointer+4, 8)))

  def test_allocation_lifecycle_is_validated(self):
    driver = QCOMDriver()
    with self.assertRaisesRegex(ValueError, 'nonzero'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0))
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=4096)
    ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, allocation)
    self.assertEqual((allocation.id, allocation.mmapsize), (1, 4096))
    ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=allocation.id))
    with self.assertRaisesRegex(ValueError, 'Unknown QCOM allocation'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=allocation.id))

  def test_contexts_and_allocations_are_descriptor_owned(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context, fd=11)
    with self.assertRaisesRegex(ValueError, 'Unknown QCOM context'):
      ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY,
            kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=context.drawctxt_id), fd=12)
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=4096)
    ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, allocation, fd=11)
    with self.assertRaisesRegex(ValueError, 'Unknown QCOM allocation'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=allocation.id), fd=12)
    driver.close(11)
    self.assertNotIn(context.drawctxt_id, driver.contexts)
    self.assertNotIn(allocation.id, driver.allocations)

  def test_pending_commands_only_block_owner_resource_release(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context, fd=11)
    driver.contexts[context.drawctxt_id].pending.append(((), 1, 0))

    allocations, mappings, backing = {}, {}, []
    for fd in (11, 12):
      allocation = kgsl.struct_kgsl_gpuobj_alloc(size=4096)
      ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, allocation, fd=fd)
      allocations[fd] = allocation
      data = (ctypes.c_uint8*16)()
      backing.append(data)
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(data), len=ctypes.sizeof(data), memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM, mapping, fd=fd)
      mappings[fd] = mapping

    with self.assertRaisesRegex(ValueError, 'pending commands'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=allocations[11].id), fd=11)
    with self.assertRaisesRegex(ValueError, 'pending commands'):
      ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=mappings[11].gpuaddr), fd=11)

    ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=allocations[12].id), fd=12)
    ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=mappings[12].gpuaddr), fd=12)

  def test_repeated_external_mapping_is_reference_counted(self):
    driver = QCOMDriver()
    data = (ctypes.c_uint8*8192)()
    pointer = ctypes.addressof(data)
    for _ in range(2):
      mapping = kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=4096, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM, mapping)
      self.assertEqual(mapping.gpuaddr, pointer)
    larger = kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=8192, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM, larger)
    self.assertEqual(driver.external[pointer].size, 8192)
    self.assertEqual(driver.external[pointer].references, 3)
    ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))
    self.assertIn(pointer, driver.external)
    self.assertEqual(driver.external[pointer].size, 8192)
    Memory(driver.ranges()).check(pointer+4096, 4096)
    ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))
    self.assertIn(pointer, driver.external)
    self.assertEqual(driver.external[pointer].size, 8192)
    Memory(driver.ranges()).check(pointer+4096, 4096)
    ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))
    self.assertNotIn(pointer, driver.external)
    with self.assertRaisesRegex(ValueError, 'Unknown QCOM user memory'):
      ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))

  def test_repeated_external_mapping_retains_canonical_extent_in_either_order(self):
    for sizes in ((4096, 8192), (8192, 4096)):
      with self.subTest(sizes=sizes):
        driver = QCOMDriver()
        data = (ctypes.c_uint8*8192)()
        pointer = ctypes.addressof(data)
        for size in sizes:
          ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
                kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))
        ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))
        self.assertEqual((driver.external[pointer].references, driver.external[pointer].size), (1, 8192))
        Memory(driver.ranges()).check(pointer+4096, 4096)
        ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer))
        self.assertNotIn(pointer, driver.external)

  def test_external_mapping_extent_and_close_are_descriptor_scoped(self):
    driver = QCOMDriver()
    data = (ctypes.c_uint8*8192)()
    pointer = ctypes.addressof(data)
    for fd,size in ((11, 8192), (12, 4096)):
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
            kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR), fd=fd)
    Memory(driver.ranges(11)).check(pointer+4096, 4096)
    with self.assertRaisesRegex(ValueError, 'outside mapped memory'): Memory(driver.ranges(12)).check(pointer+4096, 1)
    with self.assertRaisesRegex(ValueError, 'Unknown QCOM user memory'):
      ioctl(driver, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, kgsl.struct_kgsl_sharedmem_free(gpuaddr=pointer), fd=13)
    driver.close(12)
    self.assertEqual((driver.external[pointer].references, driver.external[pointer].size), (1, 8192))
    Memory(driver.ranges(11)).check(pointer+4096, 4096)
    driver.close(11)
    self.assertNotIn(pointer, driver.external)

  def test_overlapping_external_mappings_are_rejected(self):
    driver = QCOMDriver()
    data = (ctypes.c_uint8*16)()
    pointer = ctypes.addressof(data)
    ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
          kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=4, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))
    ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
          kgsl.struct_kgsl_map_user_mem(hostptr=pointer+8, len=4, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))
    for start,size in ((pointer+2, 4), (pointer, 12)):
      with self.subTest(start=start, size=size), self.assertRaisesRegex(ValueError, 'Overlapping'):
        ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
              kgsl.struct_kgsl_map_user_mem(hostptr=start, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))

  def test_external_mapping_cannot_alias_driver_allocation(self):
    driver = QCOMDriver()
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=4096)
    ioctl(driver, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, allocation, fd=11)
    pointer = driver.mmap(11, 0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE, allocation.id*0x1000)
    try:
      with self.assertRaisesRegex(ValueError, 'Overlapping'):
        ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
              kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=4096, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR), fd=12)
      self.assertNotIn(pointer, driver.external)
    finally: driver.close(11)

  def test_submission_timestamps_are_per_context(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    for context in contexts: ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context)
    command = kgsl.struct_kgsl_command_object(gpuaddr=0x1000, size=4, flags=kgsl.KGSL_CMDLIST_IB)
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), cmdsize=ctypes.sizeof(command), numcmds=1,
                                              context_id=contexts[0].drawctxt_id)
    with patch.object(driver.gpu, 'capture', return_value=()) as capture:
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND, submission)
      capture.assert_called_once_with(0x1000, 4, ())
    self.assertEqual(submission.timestamp, 1)
    timestamp = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid(context_id=contexts[1].drawctxt_id,
                                                                type=kgsl.KGSL_TIMESTAMP_QUEUED)
    ioctl(driver, kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, timestamp)
    self.assertEqual(timestamp.timestamp, 0)
    with self.assertRaisesRegex(RuntimeError, 'has not completed'):
      ioctl(driver, kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID,
            kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=contexts[1].drawctxt_id, timestamp=1))

  def test_wait_resumes_after_another_context_signals(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    for context in contexts: ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context)
    signal, result = ctypes.c_uint32(0), ctypes.c_uint32(0)
    signal_pointer, result_pointer = ctypes.addressof(signal), ctypes.addressof(result)
    control = mesa.WRITE_EQ << mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT | mesa.POLL_MEMORY << mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
    waiter_words = [pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6), control, signal_pointer & 0xffffffff, signal_pointer >> 32, 1, 0xffffffff, 0,
                    pkt7_hdr(mesa.CP_EVENT_WRITE, 4), mesa.CACHE_FLUSH_TS, result_pointer & 0xffffffff, result_pointer >> 32, 99]
    producer_words = [pkt7_hdr(mesa.CP_EVENT_WRITE, 4), mesa.CACHE_FLUSH_TS, signal_pointer & 0xffffffff, signal_pointer >> 32, 1]
    waiter, producer = (ctypes.c_uint32*len(waiter_words))(*waiter_words), (ctypes.c_uint32*len(producer_words))(*producer_words)
    for pointer,size in ((signal_pointer, 4), (result_pointer, 4), (ctypes.addressof(waiter), ctypes.sizeof(waiter)),
                         (ctypes.addressof(producer), ctypes.sizeof(producer))):
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
            kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))
    def submit(context, command):
      obj = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(command), size=ctypes.sizeof(command), flags=kgsl.KGSL_CMDLIST_IB)
      request = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1,
                                            context_id=context.drawctxt_id)
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND, request)
      return request.timestamp
    self.assertEqual(submit(contexts[0], waiter), 1)
    self.assertEqual((signal.value, result.value), (0, 0))
    self.assertEqual(driver.contexts[contexts[0].drawctxt_id].retired, 0)
    self.assertEqual(submit(contexts[1], producer), 1)
    self.assertEqual((signal.value, result.value), (1, 99))
    self.assertEqual(driver.contexts[contexts[0].drawctxt_id].retired, 1)

  def test_context_state_is_restored_across_waits(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    for context in contexts: ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context)
    signal = ctypes.c_uint32(0)
    signal_pointer = ctypes.addressof(signal)
    register = mesa.REG_A6XX_SP_CS_CONFIG
    control = mesa.WRITE_EQ << mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT | mesa.POLL_MEMORY << mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
    waiter_words = [pkt4_hdr(register, 1), 111, pkt7_hdr(mesa.CP_WAIT_REG_MEM, 6), control,
                    signal_pointer & 0xffffffff, signal_pointer >> 32, 1, 0xffffffff, 0]
    producer_words = [pkt4_hdr(register, 1), 222, pkt7_hdr(mesa.CP_EVENT_WRITE, 4), mesa.CACHE_FLUSH_TS,
                      signal_pointer & 0xffffffff, signal_pointer >> 32, 1]
    commands = [(ctypes.c_uint32*len(words))(*words) for words in (waiter_words, producer_words)]
    for pointer,size in ((signal_pointer, 4), *((ctypes.addressof(command), ctypes.sizeof(command)) for command in commands)):
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
            kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR))
    observed = []
    original_execute = driver.gpu.execute_packets
    def observe_execute(packets, start=0, ranges=None):
      observed.append(driver.gpu.registers[register])
      return original_execute(packets, start, ranges)
    with patch.object(driver.gpu, 'execute_packets', side_effect=observe_execute):
      for context,command in zip(contexts, commands):
        obj = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(command), size=ctypes.sizeof(command), flags=kgsl.KGSL_CMDLIST_IB)
        ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND,
              kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1,
                                           context_id=context.drawctxt_id))
    # The blocked context is retried once before and once after the producer runs. Each retry must restore its own state,
    # while the producer starts from its independent default state.
    self.assertEqual(observed, [0, 111, 111, 0, 111])
    self.assertEqual([driver.contexts[context.drawctxt_id].state.registers[register] for context in contexts], [111, 222])

  def test_submission_cannot_access_another_descriptors_mapping(self):
    driver = QCOMDriver()
    target = ctypes.c_uint32(0)
    target_pointer = ctypes.addressof(target)
    words = [pkt7_hdr(mesa.CP_EVENT_WRITE, 4), mesa.CACHE_FLUSH_TS, target_pointer & 0xffffffff, target_pointer >> 32, 99]
    command = (ctypes.c_uint32*len(words))(*words)
    for fd,pointer,size in ((11, target_pointer, 4), (12, ctypes.addressof(command), ctypes.sizeof(command))):
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
            kgsl.struct_kgsl_map_user_mem(hostptr=pointer, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR), fd=fd)
    context = kgsl.struct_kgsl_drawctxt_create()
    ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context, fd=12)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(command), size=ctypes.sizeof(command), flags=kgsl.KGSL_CMDLIST_IB)
    with self.assertRaisesRegex(ValueError, 'outside mapped memory'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND,
            kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1,
                                         context_id=context.drawctxt_id), fd=12)
    self.assertEqual(target.value, 0)

  def test_concurrent_drains_execute_each_submission_once(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context)
    state = driver.contexts[context.drawctxt_id]
    state.pending.append(((), 1, 0))
    gate, errors = threading.Barrier(3), []
    def execute(*_args):
      time.sleep(0.05)
      return True,0
    def drain():
      gate.wait()
      try: driver.drain()
      except Exception as error: errors.append(error)
    with patch.object(driver.gpu, 'execute_packets', side_effect=execute) as execute_packets:
      threads = [threading.Thread(target=drain) for _ in range(2)]
      for thread in threads: thread.start()
      gate.wait()
      for thread in threads: thread.join()
    self.assertEqual(errors, [])
    self.assertEqual(execute_packets.call_count, 1)
    self.assertEqual((state.consumed, state.retired, len(state.pending)), (1, 1, 0))

  def test_faulted_context_does_not_block_other_descriptors(self):
    driver = QCOMDriver()
    contexts = [kgsl.struct_kgsl_drawctxt_create() for _ in range(2)]
    commands = [(ctypes.c_uint32*2)(pkt7_hdr(mesa.CP_SET_MARKER, 1), marker)
                for marker in (0xffffffff, mesa.RM6_COMPUTE)]
    for fd,context,command in zip((11, 12), contexts, commands):
      ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context, fd=fd)
      ioctl(driver, kgsl.IOCTL_KGSL_MAP_USER_MEM,
            kgsl.struct_kgsl_map_user_mem(hostptr=ctypes.addressof(command), len=ctypes.sizeof(command),
                                          memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR), fd=fd)
    def submit(index:int):
      command = commands[index]
      obj = kgsl.struct_kgsl_command_object(gpuaddr=ctypes.addressof(command), size=ctypes.sizeof(command), flags=kgsl.KGSL_CMDLIST_IB)
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND,
            kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1,
                                         context_id=contexts[index].drawctxt_id), fd=11+index)
    with self.assertRaisesRegex(ValueError, 'Unsupported A630 marker'): submit(0)
    submit(1)
    self.assertEqual(driver.contexts[contexts[1].drawctxt_id].retired, 1)
    timestamp = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid(context_id=contexts[0].drawctxt_id, type=kgsl.KGSL_TIMESTAMP_RETIRED)
    with self.assertRaisesRegex(ValueError, 'Unsupported A630 marker'):
      ioctl(driver, kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, timestamp, fd=11)

  def test_malformed_submission_is_rejected(self):
    driver = QCOMDriver()
    context = kgsl.struct_kgsl_drawctxt_create()
    ioctl(driver, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, context)
    with self.assertRaisesRegex(ValueError, 'Invalid QCOM command list'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND, kgsl.struct_kgsl_gpu_command(context_id=context.drawctxt_id))
    command = kgsl.struct_kgsl_command_object(gpuaddr=0x1000, size=3, flags=kgsl.KGSL_CMDLIST_IB)
    submission = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), cmdsize=ctypes.sizeof(command), numcmds=1,
                                              context_id=context.drawctxt_id)
    with self.assertRaisesRegex(ValueError, 'Invalid QCOM indirect command buffer'):
      ioctl(driver, kgsl.IOCTL_KGSL_GPU_COMMAND, submission)

if __name__ == '__main__': unittest.main()
