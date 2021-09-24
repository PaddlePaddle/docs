# API Description - IO
## ReadData
### Function Definition

```
template <typename Tx, typename Ty, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(Ty* dst, const Tx* src, int size_nx, int size_ny, int stride_nx, int stride_ny);
```

### Detailed Description

Read the Tx type 2D data from the global memory to the register, and store it in the register dst according to the Ty type. Every reading of 1 column of data needs to shift the stride_nx column, and every reading of NX column of data needs to shift the stride_ny row, until NX x NY data is loaded into the register dst. When IsBoundary = true, it is necessary to ensure that the block row offset does not exceed size_ny, and the block column offset does not exceed size_nx.

### Template Parameters

> Tx: The type of data stored in the global memory. </br>
> Ty: The type of data stored in the register. </br>
> NX: Each thread reads NX column data. </br>
> NY: Each thread reads NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as thread index, and for XPU, core_id() is used as thread index. </br>
> IsBoundary: Identifies whether to fetch memory boundary judgment. When the total number of data processed by the block is less than NX x NY x blockDim.x, boundary judgment is required to avoid memory access crossing the boundary. </br>

### Parameters

> dst: output register pointer, data type is Ty, size is NX x NY. </br>
> src: The input data pointer of the current block, the data type is Tx, and the pointer calculation method is usually input + blockIdx.x x blockDim.x x NX. </br>
> size_nx: block needs to read size_nx columns data, the parameter is only used when IsBoundary = true. </br>
> size_ny: block needs to read size_ny rows data, the parameter is only used when IsBoundary = true. </br>
> stride_nx: Each column of data read needs to be offset by the stride_nx columns. </br>
> stride_ny: Each read NX column needs to be offset by stride_nx rows. </br>

------------------

## ReadData

### Function Definition

```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void ReadData(T* dst, const T* src, int num);
```

### Detailed Description

Read the 1D data of type T from the global memory src to the register dst. Continuously read NX data at a time. Currently, only NY = 1 is supported until NX data is loaded into the register dst. When IsBoundary = true, it is necessary to ensure that the total number of data read by the block does not exceed num to avoid memory fetching out of bounds. When (NX% 4 = 0 or NX% 2 = 0) and IsBoundary = false, there will be higher memory access efficiency.

### Template Parameters

> T: element type. </br>
> NX: Each thread reads NX column data. </br>
> NY: Each thread reads NY row data. Currently, only NY = 1 is supported. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as thread index, and for XPU, core_id() is used as thread index. </br>
> IsBoundary: Identifies whether to fetch memory boundary judgment. When the total number of data processed by the block is less than NX x NY x blockDim.x, boundary judgment is required to avoid memory access crossing the boundary. </br>

### Parameters

> dst: output register pointer, the size is NX x NY. </br>
> src: The input data pointer of the current block, usually input + blockIdx.x x blockDim.x x NX.</br>
> num: The current block can read at most num elements. The parameter is only used when IsBoundary = true.</br>

------------------

## ReadDataBc

### Function Definition

```
template <typename T, int NX, int NY, int BlockSize, int Rank, bool IsBoundary = false>
__device__ void ReadDataBc(T* dst, const T* src,
                           uint32_t block_offset,
                           details::BroadcastConfig<Rank> config,
                           int total_num_output,
                           int stride_nx,
                           int stride_ny);
```

### Detailed Description

Read the 2D data that needs to be brodcast from the global memory src into the register dst according to the T type, where src is the original input data pointer, calculate the input data coordinates corresponding to the current output data according to config, and read the data corresponding to the coordinates To the register.

### Template Parameters

> T: element type. </br>
> NX: Each thread reads NX column data. </br>
> NY: Each thread reads NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as thread index, and for XPU, core_id() is used as thread index. </br>
> Rank: The dimension of the original output data. </br>
> IsBoundary: Identifies whether to fetch memory boundary judgment. When the total number of data processed by the block is less than NX x NY x blockDim.x, boundary judgment is required to avoid memory access crossing the boundary. </br>

### Parameters

> dst: output register pointer, the size is NX x NY. </br>
> src: pointer to raw input data. </br>
> block_offset: The data offset of the current block, usually blockIdx.x x blockDim.x x NX. </br>
> config: Input and output coordinate mapping function, which can be defined by BroadcastConfig(const std::vector<int64_t>& out_dims, const std::vector<int64_t>& in_dims, int dim_size). </br>
> total_num_output: the total number of original output data, to avoid fetching out of bounds, the parameter is only used when IsBoundary = true. </br>
> stride_nx: Each column of data read needs to be offset by the stride_nx column. </br>
> stride_ny: Each read NX column needs to be offset by stride_nx rows. </br>


------------------

## ReadDataReduce

### Function Definition

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
### Detailed Description

Read the 2D data from the global memory SRC into the register DST in T type, where SRC is the original input data pointer according to the index_ Cal calculates the input data coordinates corresponding to the current output data and reads the data corresponding to the coordinates into the register.

### Template Parameters

> T: element type. </br>
> NX: Each thread reads NX column data. </br>
> NY: Each thread reads NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as thread index, and for XPU, core_id() is used as thread index. </br>
> Rank: The dimension of the original output data. </br>
> IndexCal: Input and output coordinate mapping rules. The definition is as follows:</br>
```
  struct IndexCal {  
    __device__ inline int operator()(int index) const {
        return ...
    }
  };
```
> IsBoundary: Identifies whether to fetch memory boundary judgment. When the total number of data processed by the block is less than NX x NY x blockDim, boundary judgment is required to avoid memory access crossing the boundary. </br>


### Parameters

> dst: output register pointer, the size is NX x NY. </br>
> src: pointer to raw input data. </br>
> block_offset: The data offset of the current block, usually blockIdx.x x blockDim.x x NX. </br>
> config: Input and output coordinate mapping function, which can be defined as IndexCal(). </br>
> size_nx: The block needs to read the size_nx column data. The parameter is only used when IsBoundary = true. </br>
> size_ny: block needs to read size_ny row data, the parameter is only used when IsBoundary = true. </br>
> stride_nx: Each column of data read needs to be offset by the stride_nx column. </br>
> stride_ny: Each read NX column needs to be offset by stride_nx row. </br>
> reduce_last_dim: Whether the lowest dimension of the original input data is reduced. When reduce_last_dim = true, it is indexed according to threadIdx.x, otherwise threadIdx.y is used. </br>

------------------

## WriteData

### Function Definition


```
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ void WriteData(T* dst, T* src, int num);
```

### Detailed Description

Write 1D data from register src to global memory dst. Continuously read NX data at a time, currently only supports NY = 1 until NX data is written to the global memory dst. When IsBoundary = true, it is necessary to ensure that the total number of data written from the current block to the world does not exceed num to avoid memory fetching out of bounds. When (NX% 4 = 0 or NX% 2 = 0) and IsBoundary = false, there will be higher memory access efficiency.

### Template Parameters

> T: element type. </br>
> NX: Each thread reads NX column data. </br>
> NY: Each thread reads NY row data. Currently, only NY = 1 is supported. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as thread index, and for XPU, core_id() is used as thread index. </br>
> IsBoundary: Identifies whether to fetch memory boundary judgment. When the total number of data processed by the block is less than NX x NY x blockDim, boundary judgment is required to avoid memory access crossing the boundary. </br>

### Parameters

> dst: The output data pointer of the current block, usually input + blockIdx.x x blockDim.x x NX. </br>
> src: register pointer, the size is NX x NY. , Usually input + blockIdx.x x blockDim.x x NX. </br>
> num: The current block reads num elements in multiples. The parameter is only used when IsBoundary = true. </br>
