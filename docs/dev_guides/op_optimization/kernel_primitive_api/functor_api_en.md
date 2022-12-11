# API Description - OpFunc
Introduce the calculation functors defined by the Kernel Primitive API. There are currently 13 functors that can be used directly.

## Unary Functor

### [ExpFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L49)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::ExpFunctor<Tx, Ty>();
```
#### Description
Exp operation is performed on the input of Tx type, and the result is converted into Ty type and returned.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
const int VecSize = 1;
float data[VecSize];
float out[VecSize];

kps::ElementwiseUnary<float, float, VecSize, 1, 1, kps::ExpFunctor<float>>(out, data, kps::ExpFunctor<float>());

// data[0] = 0;
// out[0] = exp(data);
// out[0] = 1
```


### [IdentityFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L63)
#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::IdentityFunctor<Tx, Ty>();
```

#### Description
Convert the input data of Tx type to Ty type and return.

#### Template Parameters
> Tx : The type of input. </br>
> Ty : The type of return. </br>

#### Example
```
const int VecSize = 1;
float data[VecSize];
int out[VecSize];

kps::ElementwiseUnary<float, int, VecSize, 1, 1, kps::IdentityFunctor<float, int>>(out, data, kps::IdentityFunctor<float, int>());

// data[0] = 1.3;
// out[0] = 1
```

### [DivideFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L77)
#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::DivideFunctor<Tx, Ty>(num);
```

#### Description
Divide the input data of Tx type by num, and convert the result into Ty type to return.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
const int VecSize = 1;
float data[VecSize];
float out[VecSize];
float num = 10.0;

kps::ElementwiseUnary<float, float, VecSize, 1, 1, kps::DivideFunctor<float>>(out, data, kps::DivideFunctor<float>(num));

// data[0] = 3.0
// out[0] = (3.0 / 10.0)
// out[0] = 0.3
```

### [SquareFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L112)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::SquareFunctor<Tx, Ty>();
```
#### Description
Perform Square operation on the input number of Tx type, and convert the result into Ty type to return.

#### Template Parameters
> Tx : The type of input.</br>
> Ty : The type of return.</br>

#### Example
```
const int VecSize = 1;
float data[VecSize];
float out[VecSize];

kps::ElementwiseUnary<float, float, VecSize, 1, 1, kps::SquareFunctor<float>>(out, data, kps::SquareFunctor<float>());

// data[0] = 3.0
// out[0] = (3.0 * 3.0)
// out[0] = 9.0
```

## Binary Functor

### [MinFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L128)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MinFunctor<T>();
```
#### Description
Returns the minimum of the two inputs. MinFunctor provides the initial() function for data initialization and returns the maximum value represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::MinFunctor<float>>(out, input1, input2, kps::MinFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] < input2[0] ? input1[0] : input2[0]
// out[0] = 1.0
```

### [MaxFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L140)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MaxFunctor<T>();
```
#### Description
Returns the maximum value of the two inputs. MaxFunctor provides the initial() function for data initialization, and returns the minimum value represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::MaxFunctor<float>>(out, input1, input2, kps::MaxFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] > input2[0] ? input1[0] : input2[0]
// out[0] = 3.0
```

### [AddFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L154)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::AddFunctor<T>();
```
#### Description
Returns the sum of the two inputs. AddFunctor provides the initial() function for data initialization, and returns the data 0 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::AddFunctor<float>>(out, input1, input2, kps::AddFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] + input2[0]
// out[0] = 4.0
```

### [MulFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L166)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MulFunctor<T>();
```
#### Description
Returns the product of two inputs. MulFunctor provides the initial() function for data initialization, and returns data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::MulFunctor<float>>(out, input1, input2, kps::MulFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] * input2[0]
// out[0] = 3.0
```

### [LogicalOrFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L178)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalOrFunctor<T>();
```
#### Description
Returns the result of the logical or operation of two inputs. LogicalOrFunctor provides the initial() function for data initialization, and returns false for the data represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
bool input1[VecSize];
bool input2[VecSize];
bool out[VecSize];

kps::ElementwiseBinary<bool, bool, VecSize, 1, 1, kps::LogicalOrFunctor<bool>>(out, input1, input2, kps::LogicalOrFunctor<bool>());

// input1[0] = false
// input2[0] = true
// out = input1[0] || input2[0]
// out[0] = true
```

### [LogicalAndFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L190)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalAndFunctor<T>();
```
#### Description
Returns the result of the logical and operation of the two inputs. LogicalAndFunctor provides the initial() function for data initialization, and returns true for the data represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
bool input1[VecSize];
bool input2[VecSize];
bool out[VecSize];

kps::ElementwiseBinary<bool, bool, VecSize, 1, 1, kps::LogicalAndFunctor<bool>>(out, input1, input2, kps::LogicalAndFunctor<bool>());

// input1[0] = false
// input2[0] = true
// out = input1[0] && input2[0]
// out[0] = false
```

### [SubFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L202)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::SubFunctor<T>();
```
#### Description
Two inputs are subtracted. SubFunctor provides the initial() function for data initialization, and returns the data 0 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::SubFunctor<float>>(out, input1, input2, kps::SubFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] - input2[0]
// out[0] = 2.0
```

### [DivFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L212)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::DivFunctor<T>();
```
#### Description
Two inputs are divided. DivFunctor provides the initial() function for data initialization, and returns the data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example
```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::DivideFunctor<float>>(out, input1, input2, kps::DivideFunctor<float>());

// input1[0] = 3.0
// input2[0] = 1.0
// out = input1[0] / input2[0]
// out[0] = 3.0
```

### [FloorDivFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L238)

#### Definition
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::FloorDivFunctor<T>();
```
#### Description
Divide the two inputs and return the integer part. FloorDivFunctor provides the initial() function for data initialization, and returns data 1 represented by the T type.

#### Template Parameters
> T : The type of data.

#### Example

```
const int VecSize = 1;
float input1[VecSize];
float input2[VecSize];
float out[VecSize];

kps::ElementwiseBinary<float, float, VecSize, 1, 1, kps::FloorDivFunctor<float>>(out, input1, input2, kps::FloorDivFunctor<float>());

// input1[0] = 3.0
// input2[0] = 2.0
// out = input1[0] / input2[0]
// out[0] = 1.0
```
## Functor Definition Rules
In the current calculation function, only ElementwiseAny supports setting the functor parameter as a pointer, and the functor of other calculation functions can only be set as a normal parameter.

### Common parameters
Except ElementwiseAny API, other calculation functions only support ordinary parameter transfer. For example, to realize (a + b) * c, Functor can be defined as follows:

ExampleFunctor2:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleTernaryFunctor {
   inline HOSTDEVICE T operator()(const T &input1, const T &input2, const T &input3) const {
       return ((input1 + input2) * input3);
   }
};
```

### Example

```
template<int VecSize, typename InT, typename OutT>
__global__ void ElementwiseTernaryKernel(InT *input0, InT *input1, InT * input2, OutT *output, int size) {
// global memory input pointer input0, input1, input2
auto functor = ExampleTernaryFunctor<InT>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
int data_offset = NX * blockIdx.x * blockDim.x;
int num = size - data_offset;
// if  num < NX * blockDim.x set IsBoundary = true

InT in0[NX * NY];
InT in1[NX * NY];
InT in2[NX * NY];
OutT out[NX * NY];

// each thread reads NX data continuously, and each block reads num data continuously
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(in0, input0 + data_offset, num);
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(in1, input1 + data_offset, num);
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(in2, input2 + data_offset, num);
kps::ElementwiseTernary<InT, OutT, NX, NY, BlockSize, ExampleTernaryFunctor<InT>>(out, in0, in1, in2, functor);

...

}
```

### Pointer
When defining the Functor of ElementwiseAny, you need to ensure that the parameter of the operate() function is an array pointer. For example, to realize the function: (a + b) * c + d, you can combine ElementwiseAny and Functor to complete the calculation.

ExampleAnyFunctor:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleAnyFunctor {
   inline HOSTDEVICE T operator()(const T * args) const { return ((arg[0] + arg[1]) * arg[2] + arg[3]); }
};
```
### example

```
template<int VecSize, typename InT, typename OutT>
__global__ void ElementwiseAnyKernel(InT *input0, InT *input1, InT * input2, InT * input3, OutT *output, int size) {
// global memory input pointer input0, input1, input2, input3
auto functor = ExampleAnyFunctor<InT>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
const int Arity = 4; // the pointers of inputs
int data_offset = NX * blockIdx.x * blockDim.x;
int num = size - data_offset;
// if  num < NX * blockDim.x set IsBoundary = true

InT inputs[Arity][NX * NY];
OutT out[NX * NY];

// each thread reads NX data continuously, and each block reads num data continuously
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(inputs[0], input0 + data_offset, num);
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(inputs[1], input1 + data_offset, num);
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(inputs[2], input2 + data_offset, num);
kps::ReadData<InT, NX, NY, BlockSize, IsBoundary>(inputs[3], input3 + data_offset, num);
kps::ElementwiseAny<InT, OutT, NX, NY, BlockSize, Arity, ExampleAnyFunctor<InT>>(out, inputs, functor);

...

}
```
