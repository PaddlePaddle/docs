# API 介绍 - Functor
介绍 Kernel Primitive API 定义的 Functor，当前一共有 13 个 Functor 可以直接使用。

## Unary Functor

### ExpFunctor

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
auto functor = kps::ExpFunctor<float>();
float input = 0;
float out = functor(input);

// out = exp(0)
// out = 1
```


### IdentityFunctor
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
auto functor = kps::DivideFunctor<float, double>();
float input = 3.0f;
double out = functor(input);

// out = 3.0;
```

### DivideFunctor
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
auto functor = kps::DivideFunctor<float>(10);
float input = 3.0f;
float out = functor(input);

// out = (3.0 / 10)
// out = 0.3
```

### SquareFunctor

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
auto functor = kps::SquareFunctor<float>();
float input = 3.0f;
float out = functor(input);

// out = 3.0 * 3.0
// out = 9.0
```


## Binary Functor

### MinFunctor

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
auto functor = kps::MinFunctor<float>();
float input1 = 0;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 < input2 ? input1 : input2
// out = 0
```

### MaxFunctor

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
auto functor = kps::MaxFunctor<float>();
float input1 = 0;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 > input2 ? input1 : input2
// out = 1
```

### AddFunctor

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
auto functor = kps::AddFunctor<float>();
float input1 = 1;
float input2 = 1;
float out = functor(input1, input2);

// out = input1 + input2
// out = 2
```

### MulFunctor

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
auto functor = kps::MulFunctor<float>();
float input1 = 1;
float input2 = 2;
float out = functor(input1, input2);

// out = input1 * input2
// out = 2
```

### LogicalOrFunctor

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
auto functor = kps::LogicalOrFunctor<bool>();
bool input1 = false;
bool input2 = true;
bool out = functor(input1, input2);

// out = input1 || input2
// out = true
```

### LogicalAndFunctor

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
auto functor = kps::LogicalAndFunctor<bool>();
bool input1 = false;
bool input2 = true;
bool out = functor(input1, input2);

// out = input1 && input2
// out = false
```

### SubFunctor

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
auto functor = kps::SubFunctor<float>();
float input1 = 1;
float input2 = 2;
float out = functor(input1, input2);

// out = input1 - input2
// out = 1
```

### DivFunctor

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
auto functor = kps::DivFunctor<float>();
float input1 = 1.0;
float input2 = 2.0;
float out = functor(input1, input2);

// out = input1 / input2
// out = 0.5
```

### FloorDivFunctor

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
auto functor = kps::FloorFunctor<float>();
float input1 = 1.0;
float input2 = 2.0;
float out = functor(input1, input2);

// out = input1 / input2
// out = 0
```
## Functor 定义规则
当前计算函数中仅 ElementwiseAny 支持 Functor 参数设置为指针，其他计算函数的 Functor 仅能设置为普通参数。

### 普通参数传递
除 ElementwiseAny API 外其他计算函数仅支持普通参数传递。例如需要实现 (a + b) * c 可将 Functor 定义如下：

ExampleFunctor2:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleFunctor2 {
   inline HOSTDEVICE T operator()(const T &input1, const T &input2, const T &input3) const {
       return ((input1 + input2) * input3);
   }
};
```
### 示例

```
// 全局内存输入指针 input0, input1, input2
auto functor = ExampleFunctor2<float>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
const int Arity = 3; // the pointers of inputs

int num = NX * NY * blockDim.x;
float inputs[Arity][NX * NY];
float output[NX * NY];

kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[0], input0, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[1], input1, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[2], input2, num);
kps::ElementwiseTernary<float, float, NX, NY, BlockSize, ExampleFunctor2<float>>(output, inpputs[0], inputs[1], inputs[2], functor);
// ...
```
### 指针传递
在进行 ElementwiseAny 的 Functor 定义时，需要保证 operate() 函数的参数是数组指针。例如要实现功能： (a + b) * c + d， 则可以结合 ElementwiseAny 与 Functor 完成对应计算。

ExampleFunctor1 定义:
```
namespace kps = paddle::operators::kernel_primitives;
template <typename T>
struct ExampleFunctor1 {
   inline HOSTDEVICE T operator()(const T * args) const { return ((arg[0] + arg[1]) * arg[2] + arg[3]); }
};
```
### 示例

```
// 全局内存输入指针 input0, input1, input2, input3
auto functor = ExampleFunctor1<float>();

const int NX = 4;
const int NY = 1;
const int BlockSize = 1;
const bool IsBoundary = false;
const int Arity = 4; // the pointers of inputs

int num = NX * NY * blockDim.x;
float inputs[Arity][NX * NY];
float output[NX * NY];
// read data from global memory, each thread read
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[0], input0, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[1], input1, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[2], input2, num);
kps::ReadData<float, NX, NY, BlockSize, IsBoundary>(inputs[3], input3, num);
kps::ElementwiseAny<float, float, NX, NY, BlockSize, Arity, ExampleFunctor1<float>>(output, inputs, functor);
```
