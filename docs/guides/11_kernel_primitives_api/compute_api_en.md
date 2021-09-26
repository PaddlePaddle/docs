# API Description - Compute
## [ElementwiseUnary](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L138)
### Function Definition

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void ElementwiseUnary(OutT* out, const InT* in, OpFunc compute)；
```

### Detailed Description

Calculate in according to the calculation rules in compute, and store the calculation results in the register out according to the outt type.

### Template Parameters

> InT: Type of input data. </br>
> OutT: The type stored in the out register. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> OpFunc: calculation function, defined as follows:</br>
```
   template <typename InT, typename OutT>
   struct XxxFunctor {
     HOSTDEVICE OutT operator()(const InT& a) const {
       return ...;
     }
   };
```

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> in: Input register pointer, the size is NX x NY. </br>
> compute: Calculation function, declared as OpFunc&lt;InT, OutT&gt;(). </br>

## [ElementwiseBinary](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L173)
### Function Definition

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void ElementwiseBinary(OutT* out, const InT* in1, const InT* in2, OpFunc compute)；
```

### Detailed Description

Calculate in1 and in2 according to the calculation rules in compute, and store the calculation result in the register out according to the OutT type.

### Template Parameters
> InT: Type of input data. </br>
> OutT: The type stored in the out register. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> OpFunc: calculation function, defined as follows:</br>

```
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
      return ...;
    }
  };

```

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> in1: The pointer of the left operand register, the size is NX x NY. </br>
> in2: Right operand register pointer, the size is NX x NY. </br>
> compute: The calculation object declared as OpFunc&lt;InT, OutT&gt;(). </br>

## [CycleBinary](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L291)

### Function Definition

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void CycleBinary(OutT* out, const InT* in1, const InT* in2, OpFunc compute)；
```

### Detailed Description

Calculate in1 and in2 according to the calculation rules in the compute, and store the calculation results in the register out according to the OutT type. The shape of in1 is [1, NX], and the shape of in2 is [NY, NX], realizing in1, in2 Loop calculation, the shape of out is [NY, NX].

### Template Parameters

> InT: Type of input data. </br>
> OutT: The type stored in the out register. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> OpFunc: calculation function, defined as follows:</br>
```
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
      return ...;
    }
  };

```

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> in1: The pointer of the left operand register, the size is NX. </br>
> in2: Right operand register pointer, the size is NX x NY. </br>
> compute: The calculation object declared as OpFunc&lt;InT, OutT&gt;(). </br>

## [ElementwiseTernary](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L210)

### Function Definition

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
 __device__ void ElementwiseTernary(OutT* out, const InT* in1, const InT* in2, const InT* in3, OpFunc compute)；

```

### Detailed Description

Calculate in1, in2, and in3 according to the calculation rules in compute, and store the calculation result in the register out according to the OutT type.

### Template Parameters

> InT: Type of input data. </br>
> OutT: The type stored in the out register. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> OpFunc: calculation function, defined as follows:</br>

```
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b, const InT& c) const {
      return ...;
    }
  };
```

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> in1: The register pointer of operand 1, the size is NX x NY. </br>
> in2: The register pointer of operand 2, the size is NX x NY. </br>
> in3: The register pointer of operand 3, the size is NX x NY. </br>
> compute: Declared as the calculation object of OpFunc&lt;InT, OutT&gt;(). </br>

## [ElementwiseAny](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L250)

### Function Definition

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, int Arity, class OpFunc>
__device__ void ElementwiseAny(OutT* out, InT (*ins)[NX * NY], OpFunc compute);
```

### Detailed Description

Calculate the input in ins according to the calculation rules in compute, and store the calculation result in the register out according to the OutT type. All input and output have the same dimensions.

### Template Parameters

> InT: Type of input data. </br>
> OutT: The type stored in the out register. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> Arity: The number of pointers in the pointer array ins. </br>
> OpFunc: calculation function, defined as follows:</br>
```
template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT* args) const {
      return ...;
    }
  };

```

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> ins: A pointer array composed of multiple input pointers, the size is Arity. </br>
> compute: The calculation object declared as OpFunc&lt;InT, OutT&gt;(). </br>

## [Reduce](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/kernel_primitives/compute_primitives.h#L332)

### Function Definition

```
template <typename T, int NX, int NY, int BlockSize, class ReduceFunctor, details::ReduceMode Mode>
__device__ void Reduce(T* out, const T* in, ReduceFunctor reducer, bool reduce_last_dim);
```

### Detailed Description

Reduce the input according to the reducer, the input shape is [NY, NX], when ReduceMode = kLocalMode, reduce in along the NX direction to complete the intra-thread protocol, out is [NY, 1]; when ReduceMode = kGlobalMode , Use shared memory to complete the protocol operation between threads in the block, the size of in and out are the same, both are [NY, NX].

### Template Parameters

> T: Type of input data. </br>
> NX: Each thread needs to calculate NX column data. </br>
> NY: Each thread needs to calculate NY row data. </br>
> BlockSize: Device attribute, which identifies the current device thread indexing method. For GPU, threadIdx.x is used as the thread index, and for XPU, core_id() is used as the thread index. </br>
> ReduceFunctor: Reduce calculation function, defined as follows:</br>
```
  template <typename InT>
  struct XxxFunctor {
     HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
       return ...;
     }
  };
```
> Mode : Reduce mode can be kGlobalMode or kLocalMode.

### Parameters

> out: Output register pointer, the size is NX x NY. </br>
> in: Input register pointer, the size is NX x NY. </br>
> reducer: reduction method, which can be defined using ReduceFunctor&lt;InT&gt;(). </br>
> reduce_last_dim: Indicates whether the last dimension of the original input is reduced. </br>
