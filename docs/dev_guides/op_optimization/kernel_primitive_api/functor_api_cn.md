# API 介绍 - OpFunc
介绍 Kernel Primitive API 定义的 Functor，当前一共有 13 个 Functor 可以直接使用。

## Unary Functor

### [ExpFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L49)

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::ExpFunctor<Tx, Ty>();
```
#### 功能介绍
对 Tx 类型的输入数据做 Exp 操作，并将结果转成 Ty 类型返回。

#### 模板参数
> Tx : 输入数据的类型。</br>
> Ty : 返回类型。</br>

#### 使用示例
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
#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::IdentityFunctor<Tx, Ty>();
```

#### 功能介绍
将 Tx 类型的输入数据转成 Ty 类型返回。

#### 模板参数
> Tx : 输入数据的类型。</br>
> Ty : 返回类型。</br>

#### 使用示例
```
const int VecSize = 1;
float data[VecSize];
int out[VecSize];

kps::ElementwiseUnary<float, int, VecSize, 1, 1, kps::IdentityFunctor<float, int>>(out, data, kps::IdentityFunctor<float, int>());

// data[0] = 1.3;
// out[0] = 1
```

### [DivideFunctor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/primitive/functor_primitives.h#L77)
#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::DivideFunctor<Tx, Ty>(num);
```

#### 功能介绍
将 Tx 类型的输入数据除以 num，并将结果转成 Ty 类型返回。

#### 模板参数
> Tx : 输入数据的类型。</br>
> Ty : 返回类型。</br>

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename Tx, typename Ty = Tx>
struct kps::SquareFunctor<Tx, Ty>();
```
#### 功能介绍
对 Tx 类型的输入数做 Square 操作，并将结果转成 Ty 类型返回。

#### 模板参数
> Tx : 输入数据的类型。</br>
> Ty : 返回类型。</br>

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MinFunctor<T>();
```
#### 功能介绍
返回两个输入中的最小值。MinFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的最大值。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MaxFunctor<T>();
```
#### 功能介绍
返回两个输入中的最大值。MaxFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的最小值。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::AddFunctor<T>();
```
#### 功能介绍
返回两个输入之和。AddFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 0。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::MulFunctor<T>();
```
#### 功能介绍
返回两个输入的乘积。MulFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 1。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalOrFunctor<T>();
```
#### 功能介绍
返回两个输入元素逻辑或操作后的结果。LogicalOrFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 false。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::LogicalAndFunctor<T>();
```
#### 功能介绍
返回两个输入元素逻辑与操作后的结果。LogicalAndFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 true。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::SubFunctor<T>();
```
#### 功能介绍
两个输入进行减法操作。SubFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 0。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::DivFunctor<T>();
```
#### 功能介绍
两个输入进行除法操作。DivFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 1。

#### 模板参数
> T : 数据类型。

#### 使用示例
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

#### 定义
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct kps::FloorDivFunctor<T>();
```
#### 功能介绍
两个输入进行除法操作，返回整数部分。FloorDivFunctor 提供了用于数据初始化的 initial() 函数，返回 T 类型表示的数据 1。

#### 模板参数
> T : 数据类型。

#### 使用示例

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
## Functor 定义规则
当前计算函数中仅 ElementwiseAny 支持 Functor 参数设置为指针，其他计算函数的 Functor 仅能设置为普通参数。

### 普通参数传递
除 ElementwiseAny API 外其他计算函数仅支持普通参数传递。例如需要实现 (a + b) * c 可将 Functor 定义如下：

ExampleTernaryFunctor:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleTernaryFunctor {
   inline HOSTDEVICE T operator()(const T &input1, const T &input2, const T &input3) const {
       return ((input1 + input2) * input3);
   }
};
```

### 示例

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

### 指针传递
在进行 ElementwiseAny 的 Functor 定义时，需要保证 operate() 函数的参数是数组指针。例如要实现功能： (a + b) * c + d， 则可以结合 ElementwiseAny 与 Functor 完成对应计算。

ExampleAnyFunctor 定义:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleAnyFunctor {
   inline HOSTDEVICE T operator()(const T * args) const { return ((arg[0] + arg[1]) * arg[2] + arg[3]); }
};
```
### 示例

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
