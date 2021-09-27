# API 介绍 - IO
介绍目前 Kernel Primitive API 提供的用于全局内存和寄存器进行数据交换的 API。当前实现的 IO 类 API 均是 Block 级别的多线程 API。函数内部以 blockDim.x 或 blockDim.y 进行线程索引，因此请详细阅读 API 使用规则。
## [ReadData](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/datamover_primitives.h#L121)
### 函数定义

```
template <typename Tx, typename Ty, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(Ty* dst, const Tx* src, int size_nx, int size_ny, int stride_nx, int stride_ny);
```

### 函数说明

将 Tx 类型的 2D 数据从全局内存中读取到寄存器，并按照 Ty 类型存储到寄存器 dst 中。每读取 1 列数据需要偏移 stride_nx 列数据，每读取 NX 列数据需要偏移 stride_ny 行数据，直到加载 NX * NY 个数据到寄存器 dst 中。当 IsBoundary = true 需要保证当前 Block 行偏移不超过 size_ny, 列偏移不超过 size_nx。

### 模板参数

> Tx ：数据存储在全局内存中的数据类型。</br>
> Ty ：数据存储到寄存器上的类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，当前该参数暂不支持。</br>
> IsBoundary ：标识是否进行访存边界判断。当 Block 处理的数据总数小于 NX * NY * blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst ：输出寄存器指针，数据类型为Ty, 大小为 NX * NY。</br>
> src ：当前 Block 的输入数据指针，数据类型为 Tx。</br>
> size_nx ：Block 需要读取 size_nx 列数据，参数仅在 IsBoundary = true 时使用。</br>
> size_ny ：Block 需要读取 size_ny 行数据，参数仅在 IsBoundary = true 时使用。</br>
> stride_nx ：每读取 1 列数据需要偏移 stride_nx 列。</br>
> stride_ny ：每读取 NX 列需要偏移 stride_nx 行。</br>

## [ReadData](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/datamover_primitives.h#L226)

### 函数定义

```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(T* dst, const T* src, int num);
```

### 函数说明

将 T 类型的 1D 数据从全局内存 src 中读取到寄存器 dst 中。每次连续读取 NX 个数据，当前仅支持 NY = 1，直到加载 NX 个数据到寄存器 dst 中。当 IsBoundary = true 需要保证 Block 读取的总数据个数不超过 num，以避免访存越界。当 (NX % 4 = 0 或 NX % 2 = 0) 且 IsBoundary = false 时，会有更高的访存效率。

### 模板参数

> T ：元素类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据，当前仅支持为 NY = 1。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，当前该参数暂不支持。</br>
> IsBoundary ：标识是否进行访存边界判断。当 Block 处理的数据总数小于 NX * NY * blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst : 输出寄存器指针，大小为 NX * NY。</br>
> src : 当前 Block 的输入数据指针。</br>
> num : 当前 Block 最多读取 num 个元素，参数仅在 IsBoundary = true 时使用。</br>

## [ReadDataBc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/datamover_primitives.h#L279)

### 函数定义

```
template <typename T, int NX, int NY, int BlockSize, int Rank, bool IsBoundary = false>
__device__ void ReadDataBc(T* dst, const T* src,
                           uint32_t block_offset,
                           details::BroadcastConfig<Rank> config,
                           int total_num_output,
                           int stride_nx,
                           int stride_ny);
```

### 函数说明

将需要进行 brodcast 的 2D 数据按照 T 类型从全局内存 src 中读取到寄存器 dst 中，其中 src 为原始输入数据指针，根据 config 计算当前输出数据对应的输入数据坐标，并将坐标对应的数据读取到寄存器中。

### 模板参数

> T ：元素类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，当前该参数暂不支持。</br>
> Rank ：原始输出数据的维度。</br>
> IsBoundary ：标识是否进行访存边界判断。当 Block 处理的数据总数小于 NX * NY * blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst ：输出寄存器指针，大小为 NX * NY。</br>
> src ：原始输入数据指针。</br>
> block_offset ：当前 Block的数据偏移。</br>
> config ：输入输出坐标映射函数，可通过 BroadcastConfig(const std::vector<int64_t>& out_dims, const std::vector<int64_t>& in_dims, int dim_size) 进行定义。</br>
> total_num_output ：原始输出的总数据个数,避免访存越界，参数仅在 IsBoundary = true 时使用。</br>
> stride_nx ：每读取 1 列数据需要偏移 stride_nx 列。</br>
> stride_ny ：每读取 NX 列需要偏移 stride_nx 行。</br>

## [ReadDataReduce](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/datamover_primitives.h#L337)

### 函数定义

```
template <typename T, int NX, int NY, int BlockSize, int Rank, typename IndexCal, bool IsBoundary = false>
__device__ void ReadDataReduce(T* dst,
                               const T* src,
                               int block_offset,
                               const IndexCal& index_cal,
                               int size_nx,
                               int size_ny,
                               int stride_nx,
                               int stride_ny,
                               bool reduce_last_dim);
```

### 函数说明

根据 index_cal 计算当前输出数据对应的输入数据坐标，将坐标对应的数据从全局内存 src 中读取到寄存器 dst 中。

### 模板参数

> T ：元素类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，当前该参数暂不支持。</br>
> Rank ：原始输出数据的维度。</br>
> IndexCal ：输入输出坐标映射规则。定义方式如下：</br>
```
  struct IndexCal {  
    __device__ inline int operator()(int index) const {
        return ...
    }
  };
```
> IsBoundary : 标识是否进行访存边界判断。当 Block 处理的数据总数小于 NX * NY * blockDim.x 时，需要进行边界判断以避免访存越界。</br>


### 函数参数

> dst ：输出寄存器指针，大小为 NX * NY。</br>
> src ：原始输入数据指针。</br>
> block_offset : 当前 Block 的数据偏移。</br>
> config : 输入输出坐标映射函数，可以定义为 IndexCal()。</br>
> size_nx : Block 需要读取 size_nx 列数据，参数仅在 IsBoundary = true 时使用。</br>
> size_ny : Block 需要读取 size_ny 行数据，参数仅在 IsBoundary = true 时使用。</br>
> stride_nx : 每读取 1 列数据需要偏移 stride_nx 列。</br>
> stride_ny : 每读取 NX 列需要偏移 stride_nx 行。</br>
> reduce_last_dim：原始输入数据的最低维是否进行reduce，当reduce_last_dim = true 按照 threadIdx.x 进行索引，否则使用 threadIdx.y。</br>

## [WriteData](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/datamover_primitives.h#L421)

### 函数定义


```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void WriteData(T* dst, T* src, int num);
```

### 函数说明

将 T 类型的 1D 数据从寄存器 src 写到全局内存 dst 中。每次连续读取 NX 个数据，当前仅支持NY = 1，直到写 NX 个数据到全局内存 dst 中。当 IsBoundary = true 需要保证当前 Block 向全局内从中写的总数据个数不超过 num，以避免访存越界。当 (NX % 4 = 0 或 NX % 2 = 0) 且 IsBoundary = false 时，会有更高的访存效率。

### 模板参数

> T ：元素类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据， 当前仅支持为NY = 1。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，当前该参数暂不支持。</br>
> IsBoundary ：标识是否进行访存边界判断。当 Block 处理的数据总数小于 NX * NY * blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst : 当前 Block 的输出数据指针。</br>
> src : 寄存器指针，大小为 NX * NY。</br>
> num : 当前 Block 对多读取 num 个元素，参数仅在 IsBoundary = true 时使用。</br>
