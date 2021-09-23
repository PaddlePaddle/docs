# API Examples - Add
## Add
+ Description: To complete the addition of two numbers of the same shape, the input is InT type, and the output is OutT type, and the corresponding calculation is completed according to the Functor.

### Functor Definition

```
AddFunctor:

template <typename InT, typename OutT>
struct AddFunctor {
  HOSTDEVICE OutT operator()(const InT &a, const InT &b) const { return statice<OutT>(a + b); }
};

```
### Kernel Description

VecSize means that each thread continuously reads VecSize elements. According to the relationship between the remaining elements num and the maximum number of elements processed by each thread VecSize x blockDim.x, the data processing is divided into two parts. The first part is when VecSize * blockDim. x> num indicates that the current data processing requires boundary processing, so set IsBoundary to true to avoid fetching out of bounds. Note that the Init function is used to initialize the registers arg0 and arg1 to avoid the occurrence of 0 when arg0 or arg1 is used as the denominator. Condition. Here, the sum of two numbers is completed according to the Functor. When two numbers need to be multiplied, the corresponding Functor can be directly modified, and the kernel code can be reused directly to improve development efficiency.

### Code

```
#include "kernel_primitives/kernel_primitives.h"
template<int VecSize, typename InT, typename OutT, typename Functor, bool IsBoundary>
__device__ void elementwiseImpl(InT *in0, InT * in1, OutT * out, Functor func, int num) {
  InT arg0[VecSize];
  InT arg1[VecSize];
  OutT result[VecSize];
  Init<InT, VecSize>(arg0, static_cast<OutT>(1.0f));
  Init<InT, VecSize>(arg1, static_cast<OutT>(1.0f));
  ReadData<InT, VecSize, 1, 1, IsBoundary>(arg0, in0, num);
  ReadData<InT, VecSize, 1, 1, IsBoundary>(arg1, in1, num);
  ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(result, arg0, arg1, func);
  WriteData<OutT, VecSize, 1, 1, IsBoundary>(out, result, num);
}

template<int VecSize, typename InT, typename OutT, typename Functor>
__global__ void elementwise(InT *in0, InT *in1, OutT *out, int size, Functor func) {
  int data_offset = VecSize * blockIdx.x * blockDim.x; // data offset of this block
  int stride = gridDim.x * blockDim.x * VecSize;
  for (int offset = data_offset; offset < size; offset += stride) {
    if (offset + blockDim.x * VecSize < size) {
      elementwiseImpl<VecSize, InT, OutT, Functor, false>(in0 + offset, in1 + offset, out + offset, func, size - offset);
    } else {
      elementwiseImpl<VecSize, InT, OutT, Functor, true>(in0 + offset, in1 + offset, out + offset, func, size - offset);
    }
  }
}

```
