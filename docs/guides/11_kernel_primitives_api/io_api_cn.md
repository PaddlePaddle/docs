# API 详细介绍 - IO
## ReadData
### 函数定义

```
template <typename Tx, typename Ty, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(Ty* dst, const Tx* src, int size_nx, int size_ny, int stride_nx, int stride_ny);
```

### 函数说明

将 Tx 类型的 2D 数据从全局内存中读取到寄存器，并按照 Ty 类型存储到寄存器 dst 中。每读取1列数据需要偏移 stride_nx 列数据，每读取 NX 列数据需要偏移 stride_ny 行数据，直到加载 NX x NY 个数据到寄存器 dst 中。当 IsBoundary = true 需要保证 block 行数据偏移不超过 size_ny，block 列数据偏移不超过 size_nx。

### 模板参数

> Tx ：数据存储在全局内存中的数据类型。</br>
> Ty ：数据存储到寄存器上的类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。</br>
> IsBoundary ：标识是否进行访存边界判断。当block处理的数据总数小于 NX x NY x blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst ：输出寄存器指针，数据类型为Ty, 大小为 NX x NY。</br>
> src ：当前 block 的输入数据指针，数据类型为 Tx，指针计算方式通常为 input + blockIdx.x x blockDim.x x NX。</br>
> size_nx ：block 需要读取 size_nx 列数据，参数仅在 IsBoundary=true 时使用。</br>
> size_ny ：block 需要读取 size_ny 行数据，参数仅在 IsBoundary=true 时使用。</br>
> stride_nx ：每读取 1 列数据需要偏移 stride_nx 列。</br>
> stride_ny ：每读取 NX 列需要偏移 stride_nx 行。</br>

------------------

## ReadData

### 函数定义


```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(T* dst, const T* src, int num);
```

### 函数说明

将 T 类型的 1D 数据从全局内存 src 中读取到寄存器 dst 中。每次连续读取 NX 个数据，当前仅支持 NY = 1，直到加载 NX 个数据到寄存器 dst 中。当 IsBoundary = true 需要保证 block 读取的总数据个数不超过 num，以避免访存越界。当 (NX % 4 = 0 或 NX % 2 = 0) 且 IsBoundary = false 时，会有更高的访存效率。

### 模板参数

> T ：元素类型。</br>
> NX ：每个线程读取 NX 列数据。</br>
> NY ：每个线程读取 NY 行数据，当前仅支持为NY = 1。</br>
> BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。</br>
> IsBoundary ：标识是否进行访存边界判断。当block处理的数据总数小于 NX x NY x blockDim.x 时，需要进行边界判断以避免访存越界。</br>

### 函数参数

> dst : 输出寄存器指针，大小为 NX x NY。
> src : 当前 block 的输入数据指针，通常为 input + blockIdx.x x blockDim.x x NX。
> num : 当前 block 最多读取 num 个元素，参数仅在 IsBoundary = true 时使用。

------------------

## ReadDataBc

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

T ：元素类型。

NX ：每个线程读取 NX 列数据。

NY ：每个线程读取 NY 行数据。

BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，而对于XPU，core_id() 用作线程索引。

Rank ：原始输出数据的维度。

IsBoundary ：标识是否进行访存边界判断。当block处理的数据总数小于 NX x NY x blockDim.x 时，需要进行边界判断以避免访存越界。

### 函数参数

dst ：输出寄存器指针，大小为 NX x NY。
src ：原始输入数据指针。
block_offset ：当前block的数据偏移，通常为 blockIdx.x * blockDim.x * NX。
config ：输入输出坐标映射函数，可通过 BroadcastConfig(const std::vector<int64_t>& out_dims, const std::vector<int64_t>& in_dims, int dim_size) 进行定义。
total_num_output ：原始输出的总数据个数,避免访存越界，参数仅在 IsBoundary = true 时使用。
stride_nx ：每读取 1 列数据需要偏移 stride_nx 列。
stride_ny ：每读取 NX 列需要偏移 stride_nx 行。


------------------

## ReadDataReduce

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

将需要进行 reduce 操作的 2D 数据以 T 类型从全局内存 src 中读取到寄存器dst中，其中 src 为原始输入数据指针，根据 index_cal 计算当前输出数据对应的输入数据坐标，将坐标对应的数据读取到寄存器中。

### 模板参数

T ：元素类型
NX ：每个线程读取 NX 列数据。
NY ：每个线程读取 NY 行数据。
BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，而对于XPU，core_id() 用作线程索引。
Rank ：原始输出数据的维度。
IndexCal ：输入输出坐标映射规则。定义方式如下：
  struct IndexCal {
    __device__ inline int operator()(int index) const {
        return ...
    }
  };
IsBoundary :标识是否进行访存边界判断。当block处理的数据总数小于 NX x NY x blockDim 时，需要进行边界判断以避免访存越界。


### 函数参数

dst ：输出寄存器指针，大小为 NX x NY。
src ：原始输入数据指针。
block_offset : 当前block的数据偏移，通常为 blockIdx.x * blockDim.x * NX。
config : 输入输出坐标映射函数，可以定义为IndexCal()。
size_nx : block 需要读取 size_nx 列数据，参数仅在 IsBoundary = true 时使用。
size_ny : block 需要读取 size_ny 行数据，参数仅在 IsBoundary = true 时使用。
stride_nx : 每读取 1 列数据需要偏移 stride_nx 列。
stride_ny : 每读取 NX 列需要偏移 stride_nx 行。
reduce_last_dim：原始输入数据的最低维是否进行reduce，当reduce_last_dim = true 按照 threadIdx.x 进行索引，否则使用 threadIdx.y。

------------------

## WriteData

### 函数定义


```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void WriteData(T* dst, T* src, int num);
```

### 函数说明

将 T 类型的 1D 数据从寄存器 src 写到全局内存 dst 中。每次连续读取 NX 个数据，当前仅支持NY = 1，直到写 NX 个数据到全局内存 dst 中。当 IsBoundary = true 需要保证当前 block 向全局内从中写的总数据个数不超过 num，以避免访存越界。当 (NX % 4 = 0 或 NX % 2 = 0) 且 IsBoundary = false 时，会有更高的访存效率。

### 模板参数

T ：元素类型。
NX ：每个线程读取 NX 列数据。
NY ：每个线程读取 NY 行数据， 当前仅支持为NY = 1。
BlockSize ：设备属性，标识当前设备线程索引方式。对于 GPU，threadIdx.x 用作线程索引，而对于XPU，core_id() 用作线程索引。
IsBoundary ：标识是否进行访存边界判断。当block处理的数据总数小于 NX x NY x blockDim 时，需要进行边界判断以避免访存越界。

### 函数参数

dst : 当前 block 的输出数据指针，通常为 input + blockIdx.x x blockDim.x x NX。
src : 寄存器指针，大小为 NX x NY。，通常为 input + blockIdx.x * blockDim.x * NX。
num : 当前 block 对多读取 num 个元素，参数仅在 IsBoundary = true 时使用。
