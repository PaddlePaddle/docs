# 集合通讯接口

## xccl_get_unique_id_size 【optional】

### 接口定义

```c++
C_Status (*xccl_get_unique_id_size)(size_t* size)
```

### 接口说明

获取 unique_id 对象的大小。

### 参数

size - unique_id 对象的大小，以字节为单位。

## xccl_get_unique_id 【optional】

### 接口定义

```c++
C_Status (*xccl_get_unique_id)(C_CCLRootId* unique_id)
```

### 接口说明

获取一个 unique_id 对象。

### 参数

unique_id - 插件需要填充的 unqiue_id 对象。

## xccl_comm_init_rank 【optional】

### 接口定义

```c++
C_Status (*xccl_comm_init_rank)(size_t nranks, C_CCLRootId* unique_id, size_t rank, C_CCLComm* comm)
```

### 接口说明

初始化 communicator。

### 参数

nranks - communicator 中 rank 数量。

unique_id - unique_id 对象。

rank - 本 rank 的 rank_id。

comm - communicator 对象。

## xccl_destroy_comm 【optional】

### 接口定义

```c++
C_Status (*xccl_destroy_comm)(C_CCLComm comm)
```

### 接口说明

销毁 communicator。

### 参数

## xccl_all_reduce 【optional】

### 接口定义

```c++
C_Status (*xccl_all_reduce)(void* send_buf, void* recv_buf, size_t count, C_DataType data_type, C_CCLReduceOp op, C_CCLComm comm, C_Stream stream)
```

### 接口说明

AllReduce 操作，对 comm 中所有 rank 的 send_buf 执行 op 操作后，发送到所有 rank 的 recv_buf。

### 参数

send_buf - 源数据。

recv_buf - 目的地址。

data_type - 操作的数据类型。

op - reduce 操作类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_broadcast 【optional】

### 接口定义

```c++
C_Status (*xccl_broadcast)(void* buf, size_t count, C_DataType data_type, size_t root, C_CCLComm comm, C_Stream stream)
```

### 接口说明

Broadcast 操作，将 root 节点数据广播到 comm 中其他 rank。

### 参数

buf - root 节点的源数据，其他 rank 节点的目的地址。

count - 操作的数据个数。

data_type - 操作的数据类型。

root - root 节点的 rank_id。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_reduce 【optional】

### 接口定义

```c++
C_Status (*xccl_reduce)(void* send_buf, void* recv_buf, size_t count, C_DataType data_type, C_CCLReduceOp op, C_CCLComm comm, C_Stream stream)
```

### 接口说明

Reduce 操作，对 comm 中所有 rank 的 send_buf 执行 op 操作后，发送到当前节点的 recv_buf。

### 参数

send_buf - 源数据。

recv_buf - 目的地址。

data_type - 操作的数据类型。

op - reduce 操作类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_all_gather 【optional】

### 接口定义

```c++
C_Status (*xccl_all_gather)(void* send_buf, void* recv_buf, size_t count, C_DataType data_type, C_CCLComm comm, C_Stream stream)
```

### 接口说明

AllGather 操作，将 comm 中所有 rank 的 send_buf 数据按 rank 顺序拼接后，发送到 comm 中所有 rank 的 recv_buf。

### 参数

send_buf - 源数据。

recv_buf - 目的地址。

count - 操作的数据个数。

data_type - 操作的数据类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_reduce_scatter 【optional】

### 接口定义

```c++
C_Status (*xccl_reduce_scatter)(void* send_buf, void* recv_buf, size_t count, C_DataType data_type, C_CCLReduceOp op, C_CCLComm comm, C_Stream stream)
```

### 接口说明

ReduceScatter 操作，将 comm 中所有 rank 的 send_buf 执行 op 操作后，按 rank 编号均为分散发送给 comm 中各个 rank 的 recv_buf。

### 参数

send_buf - 源数据。

recv_buf - 目的地址。

count - 操作的数据个数。

data_type - 操作的数据类型。

op - reduce 操作类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_group_start 【optional】

### 接口定义

```c++
C_Status (*xccl_group_start)()
```

### 接口说明

开始集合通迅操作聚合。

## xccl_group_end 【optional】

### 接口定义

```c++
C_Status (*xccl_group_end)()
```

### 接口说明

停止集合通迅操作聚合。

## xccl_send 【optional】

### 接口定义

```c++
C_Status (*xccl_send)(void* send_buf, size_t count, C_DataType data_type, size_t dest_rank, C_CCLComm comm, C_Stream stream)
```

### 接口说明

Send 操作，当前节点发送 send_buf 中数据到目的 rank 节点。

### 参数

send_buf - 当前 rank 的源数据。

count - 操作的数据个数。

data_type - 操作的数据类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。

## xccl_recv 【optional】

### 接口定义

```c++
C_Status (*xccl_recv)(void* recv_buf, size_t count, C_DataType data_type, size_t src_rank, C_CCLComm comm, C_Stream stream)
```

### 接口说明

Recv 操作，从 src_rank 节点接受数据到当前节点的 recv_buf。

### 参数

recv_buf - 当前节点的目的地址。

count - 操作的数据个数。

data_type - 操作的数据类型。

comm - 集合通讯操作所在 communicator。

stream - 本 rank 使用的 stream。
