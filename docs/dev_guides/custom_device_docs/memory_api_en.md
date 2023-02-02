# Memory APIs

## device_memory_allocate 【required】

### Definition

```c++
C_Status (*device_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### Description

It allocates the device memory.

### Parameter

device - the device to be used

ptr - the address of the allocated device memory

size - the size of the device memory needed to be allocated (in byte)

## device_memory_deallocate 【required】

### Definition

```c++
C_Status (*device_memory_deallocate)(const C_Device device, void* ptr, size_t size)
```

### Description

It deallocates the device storage.

### Parameter

device - the device to be used

ptr - the address of the device memory needed to be deallocated

size - the size of the device memory needed to be deallocated (in byte)

## host_memory_allocate 【optional】

### Definition

```c++
C_Status (*host_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### Description

It allocates pinned host memory.

### Parameter

device - the device to be used

ptr - the address of allocated host memory

size - the size of memory needed to be allocated (in byte)

## host_memory_deallocate 【optional】

### Definition

```c++
C_Status (*host_memory_deallocate)(const C_Device device, void* ptr, size_t size)
```

### Description

It deallocates the pinned host memory.

### Parameter

device - the device to be used

ptr - the address of host memory needed to be deallocated

size - the size of memory needed to be deallocated (in byte)

## unified_memory_allocate 【optional】

### Definition

```c++
C_Status (*unified_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### Description

It allocates unified memory.

### Parameter

device - the device to be used

ptr - unified memory address

size - the size of memory needed to be allocated (in byte)

## unified_memory_deallocate 【optional】

### Definition

```c++
C_Status (*unified_memory_deallocate)(const C_Device device, void** ptr, size_t size)
```

### Description

It deallocates unified memory.

### Parameter

device - the device to be used

ptr - the address of unified memory needed to be deallocated

size - the size of memory needed to be deallocated (in byte)

## memory_copy_h2d 【optional】

### Definition

```c++
C_Status (*memory_copy_h2d)(const C_Device device, void* dst, const void* src, size_t size)
```

### Description

It copies synchronous memory from the host to the device.

### Parameter

device - the device to be used

dst - the address of destination device memory

src - the address of the source host memory

size - the size of memory needed to be copied (in byte)

## memory_copy_d2h 【optional】

### Definition

```c++
C_Status (*memory_copy_d2h)(const C_Device device, void* dst, const void* src, size_t size)
```

### Description

It copies synchronous memory from the device to the host.

### Parameter

device - the device to be used

dst - the address of the destination host memory

src - the address of the source device memory

size - the size of memory needed to be copied (in byte)

## memory_copy_d2d 【optional】

### Definition

```c++
C_Status (*memory_copy_d2d)(const C_Device device, void* dst, const void* src, size_t size)
```

### Description

It copies synchronous memory in the device.

### Parameter

device - the device to be used

dst - the address of the destination device memroy

src - the address of the source device memory

size - the size of memory needed to be copied (in byte)

## memory_copy_p2p 【optional】

### Definition

```c++
C_Status (*memory_copy_p2p)(const C_Device dst_device, const C_Device src_device, void* dst, const void* src, size_t size)
```

### Description

It copies synchronous memory between devices.

### Parameter

dst_device - the destination device

src_device - the source device

dst - the address of destination device memory

src - the address of source device memory

size - the size of memory needed to be copied (in byte)

## async_memory_copy_h2d 【optional】

### Definition

```c++
C_Status (*async_memory_copy_h2d)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### Description

It copies asynchronous memory from the host to the device. If it is not implemented, PaddlePaddle will be replace it with a synchronous API.

### Parameter

device - the device to be used

stream - it is executed on that stream.

dst - the address of destination device memory

src - the address of source host memory

size - the size of memory neeeded to be copied (in byte)

## async_memory_copy_d2h 【optional】

### Definition

```c++
C_Status (*async_memory_copy_d2h)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### Description

It copies asynchronous memory from device to host. If it is not implemented, PaddlePaddle will replace it with a synchronous API.

### Parameter

device - the device to be used

stream - It is executed on the stream.

dst - the address of destination host

src - the address of source device

size - the size of memory needed to be copied

## async_memory_copy_d2d 【optional】

### Definition

```c++
C_Status (*async_memory_copy_d2d)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### Description

It copies asynchronous memory in the device. If it is not implemented, PaddlePaddle will replace it with a synchronous API.

### Parameter

device - the device to be used

stream - the stream to be used

dst - the address of destination device memory

src - the address of source device memory

size - the size of memory needed to be copied (in byte)

## async_memory_copy_p2p 【optional】

### Definition

```c++
C_Status (*async_memory_copy_p2p)(const C_Device dst_device, const C_Device src_device, C_Stream stream, void* dst, const void* src, size_t size)
```

### Description

It copies asynchronous memory between devices. If it is not implemented, PaddlePaddle will replace it with a synchronous API.

### Parameter

dst_device - the destination device

src_device - the source device

stream - the stream to be used

dst - the address of destination device memory

src - the address of source device memory

size - the size of memory needed to be copied (in byte)

## device_memory_set 【optional】

### Definition

```c++
C_Status (*device_memory_set)(const C_Device device, void* ptr, unsigned char value, size_t size)
```

### Description

It uses the value to pad the memory of a device. If it is not implemented, PaddlePaddle will take its place with memory_copy_h2d.

### Parameter

device - the device to be used

ptr -  the address of the padding

value - padded value

size - padding size (in byte)

## device_memory_stats 【optional】

### Definition

```c++
C_Status (*device_memory_stats)(const C_Device device, size_t* total_memory,  size_t* free_memory)
```

### Description

It counts the memory using condition.

### Parameter

device - the device to be used

total_memory - total memory (in byte)

free_memory - free memory (in byte)

## device_min_chunk_size 【optional】

### Definition

```c++
C_Status (*device_min_chunk_size)(C_Device device, size_t* size)
```

### Description

It checks the minimum size of device memory chunks (in byte). In order not to call the device API to frequently apply for/ deallocate memory, PaddlePaddle manages the device memory. When there is an application, memory will be first allocated from the managed memory. When there is an application for memory whose size is "size", the size of the allocated memory is size + extra_padding_size and it will be aligned with min_chunk_size, the minimum size of memory chunks.

### Parameter

device - the device to be used

size - the size of the minimum chunk (in byte)

## device_max_chunk_size 【optional】

### Definition

```c++
C_Status (*device_max_chunk_size)(C_Device device, size_t* size)
```

### Description

The size of the memory allocated from that managed by PaddlePaddle is no more than the maximum size of device memory chunks (in byte). Otherwise, the device API will be invoked for allocation. If this API is not implemented, the size of the memory is device_max_alloc_size, the maximum size of allocatable device memory.

### Parameter

device - the device to be used

size - the size of the maximum chunk (in byte)

## device_max_alloc_size 【optional】

### Definition

```c++
C_Status (*device_max_alloc_size)(C_Device device, size_t* size)
```

### Description

It checks the maximum size (in byte) of allocatable device memory. If it is not implemented, the memory size will be equal to that of the current available memory.

### Parameter

device - the device to be used

size - the maximum size of allocatable memory (in byte)

## device_extra_padding_size 【optional】

### Definition

```c++
C_Status (*device_extra_padding_size)(C_Device device, size_t* size)
```

### Description

It allocates the extra padding size of device memory. If it is not implemented, the size will be set to 0 by default. In order not to call the device API to frequently apply for/ deallocate memory, PaddlePaddle manages the device memory. When there is an application, memory will be first allocated from the managed memory. When there is an application for memory whose size is "size", the size of the allocated memory is size + extra_padding_size and it will be aligned with min_chunk_size, the minimum size of memory chunks.

### Parameter

device - the device to be used

size - the extra padding size (in byte)

## device_init_alloc_size 【optional】

### Definition

```c++
C_Status (*device_init_alloc_size)(const C_Device device, size_t* size)
```

### Description

It checks the size of allocated device memory (in byte) after initialization. If it is not implemented, the size will be equal to device_max_alloc_size, the maximum size of allocatable device memory.

### Parameter

device - the device to be used

size - the size of first allocated memory (in byte)

## device_realloc_size 【optional】

### Definition

```c++
C_Status (*device_realloc_size)(const C_Device device, size_t* size)
```

### Description

It checks the size of reallocated device memory (in byte). If it is not implemented, the memory size will be equal to device_max_alloc_size, the maximum size of allocatable device memory.

### Parameter

device - the device to be used

size - the size of reallocated memory (in byte)
