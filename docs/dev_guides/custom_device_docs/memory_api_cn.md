# Memory 接口

## device_memory_allocate 【required】

### 接口定义

```c++
C_Status (*device_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### 接口说明

分配设备内存。

### 参数

device - 使用的设备。

ptr - 存储分配的设备内存地址。

size - 需要分配的设备内存大小（字节形式）。

## device_memory_deallocate 【required】

### 接口定义

```c++
C_Status (*device_memory_deallocate)(const C_Device device, void* ptr, size_t size)
```

### 接口说明

释放设备内存。

### 参数

device - 使用的设备。

ptr - 需要释放的设备内存地址。

size - 需要释放的设备内存大小（字节形式）。

## host_memory_allocate 【optional】

### 接口定义

```c++
C_Status (*host_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### 接口说明

分配主机锁页内存。

### 参数

device - 使用的设备。

ptr - 存储分配的主机内存地址。

size - 需要分配的内存大小（字节形式）。

## host_memory_deallocate 【optional】

### 接口定义

```c++
C_Status (*host_memory_deallocate)(const C_Device device, void* ptr, size_t size)
```

### 接口说明

释放主机锁页内存。

### 参数

device - 使用的设备。

ptr - 需要释放的主机内存地址。

size - 需要释放的内存大小（字节形式）。

## unified_memory_allocate 【optional】

### 接口定义

```c++
C_Status (*unified_memory_allocate)(const C_Device device, void** ptr, size_t size)
```

### 接口说明

分配统一地址空间内存。

### 参数

device - 使用的设备。

ptr - 存储分配的统一地址空间内存地址。

size - 需要分配内存的大小（字节形式）。

## unified_memory_deallocate 【optional】

### 接口定义

```c++
C_Status (*unified_memory_deallocate)(const C_Device device, void** ptr, size_t size)
```

### 接口说明

释放统一地址空间内存。

### 参数

device - 使用的设备。

ptr - 需要释放的统一地址空间内存地址。

size - 需要释放的内存大小（字节形式）。

## memory_copy_h2d 【optional】

### 接口定义

```c++
C_Status (*memory_copy_h2d)(const C_Device device, void* dst, const void* src, size_t size)
```

### 接口说明

主机到设备的同步内存拷贝。

### 参数

device - 使用的设备。

dst - 目的设备内存地址。

src - 源主机内存地址。

size - 需要拷贝的内存大小（字节形式）。

## memory_copy_d2h 【optional】

### 接口定义

```c++
C_Status (*memory_copy_d2h)(const C_Device device, void* dst, const void* src, size_t size)
```

### 接口说明

设备到主机的同步内存拷贝。

### 参数

device - 使用的设备。

dst - 目的主机内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小（字节形式）。

## memory_copy_d2d 【optional】

### 接口定义

```c++
C_Status (*memory_copy_d2d)(const C_Device device, void* dst, const void* src, size_t size)
```

### 接口说明

设备内同步内存拷贝。

### 参数

device - 使用的设备。

dst - 目的设备内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小（字节形式）。

## memory_copy_p2p 【optional】

### 接口定义

```c++
C_Status (*memory_copy_p2p)(const C_Device dst_device, const C_Device src_device, void* dst, const void* src, size_t size)
```

### 接口说明

设备间同步内存拷贝。

### 参数

dst_device - 目的设备。

src_device - 源设备。

dst - 目的设备内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小（字节形式）。

## async_memory_copy_h2d 【optional】

### 接口定义

```c++
C_Status (*async_memory_copy_h2d)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### 接口说明

主机到设备的异步内存拷贝，如果没有实现，PaddlePaddle 会用同步接口代替。

### 参数

device - 使用的设备。

stream - 在该 stream 上执行。

dst - 目的设备内存地址。

src - 源主机内存地址。

size - 需要拷贝的内存大小（字节形式）。

## async_memory_copy_d2h 【optional】

### 接口定义

```c++
C_Status (*async_memory_copy_d2h)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### 接口说明

设备到主机的异步内存拷贝，如果没有实现，PaddlePaddle 会用同步接口代替。

### 参数

device - 使用的设备。

stream - 在该 stream 上执行。

dst - 目的主机内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小。

## async_memory_copy_d2d 【optional】

### 接口定义

```c++
C_Status (*async_memory_copy_d2d)(const C_Device device, C_Stream stream, void* dst, const void* src, size_t size)
```

### 接口说明

设备内异步内存拷贝，如果没有实现，PaddlePaddle 会用同步接口代替。

### 参数

device - 使用的设备。

stream - 使用的 stream。

dst - 目的设备内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小（字节形式）。

## async_memory_copy_p2p 【optional】

### 接口定义

```c++
C_Status (*async_memory_copy_p2p)(const C_Device dst_device, const C_Device src_device, C_Stream stream, void* dst, const void* src, size_t size)
```

### 接口说明

设备间异步内存拷贝，如果没有实现，PaddlePaddle 会用同步接口代替。

### 参数

dst_device - 目的设备。

src_device - 源设备。

stream - 使用的 stream。

dst - 目的设备内存地址。

src - 源设备内存地址。

size - 需要拷贝的内存大小（字节形式）。

## device_memory_set 【optional】

### 接口定义

```c++
C_Status (*device_memory_set)(const C_Device device, void* ptr, unsigned char value, size_t size)
```

### 接口说明

使用 value 填充一块设备内存，如果没有实现，PaddlePaddle 会用 memory_copy_h2d 代替。

### 参数

device - 使用的设备。

ptr - 填充地址。

value - 填充值。

size - 填充大小（字节形式）。

## device_memory_stats 【optional】

### 接口定义

```c++
C_Status (*device_memory_stats)(const C_Device device, size_t* total_memory,  size_t* free_memory)
```

### 接口说明

设备内存使用统计。

### 参数

device - 使用的设备。

total_memory - 总内存（字节形式）。

free_memory - 剩余可用内存（字节形式）。

## device_min_chunk_size 【optional】

### 接口定义

```c++
C_Status (*device_min_chunk_size)(C_Device device, size_t* size)
```

### 接口说明

获取设备内存的最小快大小（字节形式）。为避免频繁调用硬件 API 申请/释放内存， PaddlePaddle 会自行管理设备内存，申请内存时优先从 PaddlePaddle 管理的内存中分配。申请 size 大小的内存时，会分配 size + extra_padding_size 大小的内存，并按 min_chunk_size 对齐。

### 参数

device - 使用的设备。

size - 最小块的大小（字节形式）。

## device_max_chunk_size 【optional】

### 接口定义

```c++
C_Status (*device_max_chunk_size)(C_Device device, size_t* size)
```

### 接口说明

PaddlePaddle 管理的设备内存一次最多分配该大小（字节形式），超过该大小时，将直接调用硬件 API 进行分配，如果没有实现，则大小等于 device_max_alloc_size。

### 参数

device - 使用的设备。

size - 最大块的大小（字节形式）。

## device_max_alloc_size 【optional】

### 接口定义

```c++
C_Status (*device_max_alloc_size)(C_Device device, size_t* size)
```

### 接口说明

设备最多可分配的内存大小（字节形式），如果没有实现，则大小等于目前可用的内存。

### 参数

device - 使用的设备。

size - 最多可分配的内存的大小（字节形式）。

## device_extra_padding_size 【optional】

### 接口定义

```c++
C_Status (*device_extra_padding_size)(C_Device device, size_t* size)
```

### 接口说明

分配设备内存需要的额外填充字节，如果没有实现，则默认为 0。为避免频繁调用硬件 API 申请/释放内存， PaddlePaddle 会自行管理设备内存，申请内存时优先从 PaddlePaddle 管理的内存中分配。申请 size 大小的内存时，会分配 size + extra_padding_size 大小的内存，并按 min_chunk_size 对齐。

### 参数

device - 使用的设备。

size - 额外填充的大小（字节形式）。

## device_init_alloc_size 【optional】

### 接口定义

```c++
C_Status (*device_init_alloc_size)(const C_Device device, size_t* size)
```

### 接口说明

设备初始分配的内存大小（字节形式），如果没有实现，则大小等于 device_max_alloc_size。

### 参数

device - 使用的设备。

size - 初始分配的内存大小（字节形式）。

## device_realloc_size 【optional】

### 接口定义

```c++
C_Status (*device_realloc_size)(const C_Device device, size_t* size)
```

### 接口说明

设备重分配的内存大小（字节形式），如果没有实现，则大小等于 device_max_alloc_size。

### 参数

device - 使用的设备。

size - 重分配的内存大小（字节形式）。
