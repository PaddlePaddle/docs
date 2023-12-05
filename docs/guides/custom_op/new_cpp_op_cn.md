# 自定义 C++算子

## 概述

算子（Operator，简称 Op）是构建神经网络的基础组件，飞桨框架提供了丰富的算子库，能够满足绝大多数场景的使用需求。但是出于以下几点原因，您可能希望定制化算子的 C++实现，从而满足特定需求：

1. 已有的算子无法组合出您需要的运算逻辑；
2. 使用已有算子组合得到的运算逻辑无法满足您的性能需求。

为此，我们提供了自定义外部算子的机制，以此机制实现的自定义算子，能够以 **即插即用** 的方式用于模型训练与推理，不需要重新编译安装飞桨框架。

使用自定义算子机制，仅需要以下两个步骤：

1. 实现算子的 C++运算逻辑，完成算子构建
2. 调用 `python` 接口完成算子编译与注册

随后即可在模型中使用，下面通过实现一个 `relu` 运算，介绍具体的实现、编译与应用流程。

> 注意事项：
> - 在使用本机制实现自定义算子之前，请确保已经正确安装了 `PaddlePaddle 2.3` 及以上版本
> - 该机制已支持 `Linux` 、 `Mac`  和 `Windows` 平台。
> - 本自定义外部算子机制仅保证源码级别的兼容，不保证二进制级别的兼容，例如，基于飞桨 2.3 版本编写的自定义算子源码实现，在飞桨 2.3 或者后续版本中编译链接使用没有问题，但基于飞桨 2.3 之前的版本编译得到的自定义算子动态库文件（*.so, *.dylib, *.dll），在 2.3 或者后续发布的版本中可能会加载失败。

## 自定义算子 C++实现

使用自定义算子机制，需要编写以下组件的 C++实现，包括：

1. **算子的运算函数**：算子核心的计算逻辑实现，主要是对输入 `Tensor` 进行处理，得到输出 `Tensor` 的过程
2. **算子的维度与类型推导函数**：用于在组网编译和运行时，正确推导出输出 `Tensor` 的 `shape` 和 `data type`
3. **算子构建**：描述算子的输入输出信息、并关联前述运算、维度推导与类型推导函数

下面结合示例进行介绍。

### 运算函数与基础 API

#### 基本写法要求

在编写运算函数之前，需要引入 `PaddlePaddle` 扩展头文件，示例如下：

```c++
#include "paddle/extension.h"
```

算子运算函数有特定的函数写法要求，在编码过程中需要遵守，基本形式如下：

```c++
std::vector<paddle::Tensor> OpFucntion(const paddle::Tensor& x, ..., int attr, ...) {
  ...
}
```

- 函数输入参数可以是 `paddle::Tensor` , `std::vector<paddle::Tensor>` 或者一些基础类型的 `Attribute` ，具体地：
    - `paddle::Tensor` 需要以 `const paddle::Tensor& ` 的形式作为输入，可以有一个或多个
    - `std::vector<paddle::Tensor>` 需要以 `const std::vector<paddle::Tensor>& ` 的形式作为输入，可以有一个或多个
    - `Attribute` 目前仅支持如下数据类型，建议按如下形式作为输入，可以有一个或多个：
        - `bool`
        - `int`
        - `float`
        - `int64_t`
        - `const std::string&`
        - `const std::vector<int>&`
        - `const std::vector<float>&`
        - `const std::vector<int64_t>&`
        - `const std::vector<std::string>&`
- 函数返回值只能是 `std::vector<paddle::Tensor>`

> 注：其他类型的数值作为函数输入参数或者返回值将无法编译通过

#### 设备类型

设备类型使用 `Place` 表示，`Place` 含有内存类型 AllocationType 与设备 ID 信息，是 `Tensor` 的基础描述信息之一。

其中设备类型是枚举类型：

```c++
enum class AllocationType : int8_t {
  UNDEFINED = 0,
  CPU = 1,
  GPU = 2,
  GPUPINNED = 3,
  ...
};
```

设备 ID 是一个 int8_t 的数值，用于表示当前使用的设备卡号。

一些 Place 使用示例如下：

```c++
auto cpu_place = paddle::CPUPlace();
auto gpu_place = paddle::GPUPlace(); // 默认设备 ID 为 0，一般在自定义算子内使用默认的构造方式即可
auto gpu_place = paddle::GPUPlace(1); // GPU 1 号卡
```

此外，Place 还有两个常用的方法：

- GetType()：获取 Place 的内存类型 AllocationType
- GetDeviceId()：获取 Place 的设备 ID

使用示例如下：

```c++
auto gpu_place = paddle::GPUPlace();
auto alloc_type = gpu_place.GetType(); // paddle::AllocationType::GPU
auto dev_id = gpu_place.GetDeviceId(); // 0
```

详细的 Place 定义请参考 [paddle/phi/common/place.h](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/paddle/phi/common/place.h)。

> 注：目前自定义算子仅在 CPU 与 GPU 上进行了验证，其他类型会视需求在后续版本支持

#### 数据类型

数据类型使用 `DataType` 表示，同样是 `Tensor` 的基础描述信息之一，目前主要支持的类型如下：

```c++
enum class DataType {
  UNDEFINED = 0,
  BOOL,
  INT8,
  UINT8,
  INT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  BFLOAT16,
  FLOAT16,
  UINT16,
  FLOAT32,
  FLOAT64,
  COMPLEX64,
  COMPLEX128,
  ...
}
```

详细的 DataType 定义请参考 [paddle/phi/common/data_type.h](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/paddle/phi/common/data_type.h)。

#### Tensor API

（1） `Tensor` 构造

对于 `paddle::Tensor` 的构造，我们推荐使用相应的初始化 paddle API，包括：

```c++
PADDLE_API Tensor empty(const IntArray& shape, DataType dtype=DataType::FLOAT32, const Place& place=CPUPlace());
PADDLE_API Tensor full(const IntArray& shape, const Scalar& value, DataType dtype=DataType::FLOAT32, const Place& place=CPUPlace());

PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype=DataType::UNDEFINED, const Place& place={});
PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype=DataType::UNDEFINED, const Place& place={});
```

使用示例如下：

```c++
auto tensor = paddle::empty({3, 4}); // default: float32, cpu
auto tensor = paddle::full({3, 4}, 1.0); // default: float32, cpu
auto gpu_tensor = paddle::empty({3, 4}, paddle::DataType::FLOAT64, paddle::GPUPlace());
auto gpu_tensor = paddle::full({3, 4}, 1.0, paddle::DataType::FLOAT64, paddle::GPUPlace());
```

（2） `Tensor` 成员方法

此外 `paddle::Tensor` 自身提供了一些基础的功能 API，常用的包括：

- 设备、数据类型获取 API：
    - `const Place& place() const`：获取 `Tensor` 所在的设备
    - `DataType dtype() const`：获取 `Tensor` 的数据类型
- 长度与维度获取 API：
    - `int64_t numel() const`：获取 `Tensor` 的数据长度
    - `std::vector<int64_t> shape() const`：获取 `Tensor` 的维度信息
- 数据访问 API：
    - `template <typename T> const T* data() const`：模板类方法，获取数据内存的起始地址（只读）
    - `template <typename T> T* data()`：模板类方法，获取数据内存的起始地址（读写）
- 状态或属性判断 API：
    - `bool defined() const`: 确认 `Tensor` 是否有效
    - `bool initialized() const`: 确认 `Tensor` 是否已被初始化
    - `bool is_cpu() const`：确认 `Tensor` 是否在 CPU 上
    - `bool is_gpu() const`：确认 `Tensor` 是否在 GPU 上
- 工具类 API：
    - `Tensor copy_to(const Place& place, bool blocking) const`：
        - 模板类方法，输入参数 `place`，将当前 `Tensor` 拷贝到指定设备上并返回
    - `Tensor cast(DataType target_type) const`：
        - 输入参数 `target_type` ，将当前 `Tensor` 转换为指定数据类型的 `Tensor` 并返回
    - `Tensor slice(const int64_t begin_idx, const int64_t end_idx) const`：
        - 输入参数起始行 begin_idx 和终止行 end_idx，返回当前 Tensor 从起始行（含）到终止行（不含）的一个视图
        - 目前仅支持对当前 Tensor 的第一个维度（即 axis = 0）进行切分
    - `cudaStream_t stream() const`：
        - 用于获取当前 `Tensor` 所处的 CUDA Stream（仅在 GPU 编译版本中生效）
        - 仅能够获取函数输入 `Tensor` 的 stream

后续我们会继续扩展其他 Tensor API，详细的 Tensor 定义请参考 [paddle/phi/api/include/tensor.h](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/paddle/phi/api/include/tensor.h) 。

#### Exception API

- `PD_CHECK(COND, ...)`：输入 bool 条件表达式进行检查，如果值为 `false` ，则抛出异常，支持变长参数输入，伪代码示例如下：

```c++
// case 1: No error message specified
PD_CHECK(a > b)
// The key error message like:
// Expected a > b, but it is not satisfied.
//   [/User/custom_op/custom_relu_op.cc:82]

// case 2: Error message specified
PD_CHECK(a > b, "PD_CHECK returns ", false, ", expected a > b.")
// The key error message like:
// PD_CHECK returns returns false, expected a > b.
//   [/User/custom_op/custom_relu_op.cc:82]
```

- `PD_THROW`：用于直接抛出异常，支持变长参数输入

```c++
// case 1: No error message specified
PD_THROW()
// The key error message like:
// An error occurred.
//   [/User/custom_op/custom_relu_op.cc:82]

// case 2: Error message specified
PD_THROW("PD_THROW returns ", false)
// The key error message like:
// PD_THROW returns false
//   [/User/custom_op/custom_relu_op.cc:82]
```

#### 类 Python 的 C++运算 API

自 paddle 2.3 版本开始，我们提供定义与用法与相应 Python API 类似的 C++ API，其 API 命名、参数顺序及类型均和相应的 paddle Python API 对齐，可以通过查找相应 Python API 的官方文档了解其用法，并在自定义算子开发时使用。通过调用这些接口，可以省去封装基础运算的时间，从而提高开发效率。

在 2.3 版本支持的 C++ API 列表如下，可以通过 `paddle::xxx` 进行调用：

```c++
PADDLE_API Tensor abs(const Tensor& x);
PADDLE_API Tensor acos(const Tensor& x);
PADDLE_API Tensor acosh(const Tensor& x);
PADDLE_API Tensor add(const Tensor& x, const Tensor& y);
PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);
PADDLE_API Tensor allclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan);
PADDLE_API std::tuple<Tensor,Tensor> argsort(const Tensor& x, int axis, bool descending);
PADDLE_API Tensor asin(const Tensor& x);
PADDLE_API Tensor asinh(const Tensor& x);
PADDLE_API Tensor atan(const Tensor& x);
PADDLE_API Tensor atan2(const Tensor& x, const Tensor& y);
PADDLE_API Tensor atanh(const Tensor& x);
PADDLE_API Tensor bernoulli(const Tensor& x);
PADDLE_API Tensor ceil(const Tensor& x);
PADDLE_API Tensor cholesky(const Tensor& x, bool upper);
PADDLE_API Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper);
PADDLE_API Tensor clip(const Tensor& x, const Scalar& min, const Scalar& max);
PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis);
PADDLE_API Tensor conj(const Tensor& x);
PADDLE_API Tensor cos(const Tensor& x);
PADDLE_API Tensor cosh(const Tensor& x);
PADDLE_API Tensor cross(const Tensor& x, const Tensor& y, int axis=9);
PADDLE_API Tensor det(const Tensor& x);
PADDLE_API Tensor diag(const Tensor& x, int offset, float padding_value);
PADDLE_API Tensor diagonal(const Tensor& x, int offset, int axis1, int axis2);
PADDLE_API Tensor digamma(const Tensor& x);
PADDLE_API Tensor dist(const Tensor& x, const Tensor& y, float p);
PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);
PADDLE_API Tensor dot(const Tensor& x, const Tensor& y);
PADDLE_API Tensor elu(const Tensor& x, float alpha);
PADDLE_API Tensor empty(const IntArray& shape, DataType dtype=DataType::FLOAT32, const Place& place=CPUPlace());
PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype=DataType::UNDEFINED, const Place& place={});
PADDLE_API Tensor equal(const Tensor& x, const Tensor& y);
PADDLE_API Tensor equal_all(const Tensor& x, const Tensor& y);
PADDLE_API Tensor erf(const Tensor& x);
PADDLE_API Tensor erfinv(const Tensor& x);
PADDLE_API Tensor exp(const Tensor& x);
PADDLE_API Tensor expand(const Tensor& x, const IntArray& shape);
PADDLE_API Tensor expm1(const Tensor& x);
PADDLE_API std::tuple<Tensor,Tensor> flatten(const Tensor& x, int start_axis, int stop_axis);
PADDLE_API Tensor flip(const Tensor& x, const std::vector<int>& axis);
PADDLE_API Tensor floor(const Tensor& x);
PADDLE_API Tensor floor_divide(const Tensor& x, const Tensor& y);
PADDLE_API Tensor fmax(const Tensor& x, const Tensor& y);
PADDLE_API Tensor fmin(const Tensor& x, const Tensor& y);
PADDLE_API Tensor frame(const Tensor& x, int frame_length, int hop_length, int axis = -1);
PADDLE_API Tensor full(const IntArray& shape, const Scalar& value, DataType dtype=DataType::FLOAT32, const Place& place=CPUPlace());
PADDLE_API Tensor gather(const Tensor& x, const Tensor& index, const Scalar& axis=0);
PADDLE_API Tensor gather_nd(const Tensor& x, const Tensor& index);
PADDLE_API Tensor gelu(const Tensor& x, bool approximate);
PADDLE_API Tensor greater_equal(const Tensor& x, const Tensor& y);
PADDLE_API Tensor greater_than(const Tensor& x, const Tensor& y);
PADDLE_API Tensor gumbel_softmax(const Tensor& x, float temperature, bool hard, int axis);
PADDLE_API Tensor hardswish(const Tensor& x);
PADDLE_API Tensor hardtanh(const Tensor& x, float t_min, float t_max);
PADDLE_API Tensor imag(const Tensor& x);
PADDLE_API Tensor increment(const Tensor& x, float value);
PADDLE_API Tensor index_sample(const Tensor& x, const Tensor& index);
PADDLE_API Tensor is_empty(const Tensor& x);
PADDLE_API Tensor isclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan);
PADDLE_API Tensor isfinite(const Tensor& x);
PADDLE_API Tensor isinf(const Tensor& x);
PADDLE_API Tensor isnan(const Tensor& x);
PADDLE_API Tensor kron(const Tensor& x, const Tensor& y);
PADDLE_API std::tuple<Tensor,Tensor> kthvalue(const Tensor& x, int k, int axis, bool keepdim);
PADDLE_API Tensor label_smooth(const Tensor& label, paddle::optional<const Tensor&> prior_dist, float epsilon);
PADDLE_API Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& weight);
PADDLE_API Tensor less_equal(const Tensor& x, const Tensor& y);
PADDLE_API Tensor less_than(const Tensor& x, const Tensor& y);
PADDLE_API Tensor lgamma(const Tensor& x);
PADDLE_API Tensor log(const Tensor& x);
PADDLE_API Tensor log10(const Tensor& x);
PADDLE_API Tensor log1p(const Tensor& x);
PADDLE_API Tensor log2(const Tensor& x);
PADDLE_API Tensor logit(const Tensor& x, float eps=1e-6f);
PADDLE_API Tensor masked_select(const Tensor& x, const Tensor& mask);
PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x=false, bool transpose_y=false);
PADDLE_API Tensor matrix_power(const Tensor& x, int n);
PADDLE_API Tensor maximum(const Tensor& x, const Tensor& y);
PADDLE_API Tensor maxout(const Tensor& x, int groups, int axis);
PADDLE_API std::vector<Tensor> meshgrid(const std::vector<Tensor>& inputs);
PADDLE_API Tensor minimum(const Tensor& x, const Tensor& y);
PADDLE_API std::tuple<Tensor,Tensor> mode(const Tensor& x, int axis, bool keepdim);
PADDLE_API Tensor multi_dot(const std::vector<Tensor>& x);
PADDLE_API Tensor multinomial(const Tensor& x, int num_samples, bool replacement);
PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);
PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec);
PADDLE_API std::tuple<Tensor,Tensor> nll_loss(const Tensor& input, const Tensor& label, paddle::optional<const Tensor&> weight, int64_t ignore_index, const std::string& reduction);
PADDLE_API Tensor not_equal(const Tensor& x, const Tensor& y);
PADDLE_API Tensor one_hot(const Tensor& x, const Scalar& num_classes);
PADDLE_API Tensor ones(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());
PADDLE_API Tensor overlap_add(const Tensor& x, int hop_length, int axis = -1);
PADDLE_API Tensor pixel_shuffle(const Tensor& x, int upscale_factor, const std::string& data_format);
PADDLE_API Tensor poisson(const Tensor& x);
PADDLE_API Tensor put_along_axis(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce = "assign", include_self=True, broadcast=True);
PADDLE_API std::tuple<Tensor,Tensor> qr(const Tensor& x, const std::string& mode);
PADDLE_API Tensor real(const Tensor& x);
PADDLE_API Tensor reciprocal(const Tensor& x);
PADDLE_API Tensor relu(const Tensor& x);
PADDLE_API Tensor relu6(const Tensor& x);
PADDLE_API Tensor remainder(const Tensor& x, const Tensor& y);
PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape);
PADDLE_API Tensor roll(const Tensor& x, const IntArray& shifts, const std::vector<int64_t>& axis);
PADDLE_API Tensor round(const Tensor& x);
PADDLE_API Tensor rsqrt(const Tensor& x);
PADDLE_API Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite);
PADDLE_API Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates);
PADDLE_API Tensor selu(const Tensor& x, float scale, float alpha);
PADDLE_API Tensor send_u_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op = "SUM", const IntArray& out_size = {0});
PADDLE_API Tensor send_ue_recv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op, const std::string& reduce_op, const IntArray& out_size);
PADDLE_API Tensor send_uv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD");
PADDLE_API Tensor sign(const Tensor& x);
PADDLE_API Tensor silu(const Tensor& x);
PADDLE_API Tensor sin(const Tensor& x);
PADDLE_API Tensor sinh(const Tensor& x);
PADDLE_API std::vector<Tensor> split(const Tensor& x, const IntArray& num_or_sections, const Scalar& axis);
PADDLE_API Tensor sqrt(const Tensor& x);
PADDLE_API Tensor square(const Tensor& x);
PADDLE_API Tensor stack(const std::vector<Tensor>& x, int axis);
PADDLE_API Tensor strided_slice(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides);
PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);
PADDLE_API Tensor swish(const Tensor& x);
PADDLE_API Tensor tanh(const Tensor& x);
PADDLE_API Tensor thresholded_relu(const Tensor& x, float threshold);
PADDLE_API Tensor tile(const Tensor& x, const IntArray& repeat_times);
PADDLE_API Tensor trace(const Tensor& x, int offset, int axis1, int axis2);
PADDLE_API Tensor triangular_solve(const Tensor& x, const Tensor& y, bool upper, bool transpose, bool unitriangular);
PADDLE_API Tensor tril(const Tensor& x, int diagonal);
PADDLE_API std::vector<Tensor> unbind(const Tensor& input, int axis);
PADDLE_API std::tuple<Tensor,Tensor,Tensor,Tensor> unique(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype=DataType::INT64);
PADDLE_API std::tuple<Tensor,Tensor> unsqueeze(const Tensor& x, const IntArray& axis);
PADDLE_API Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);
PADDLE_API Tensor zeros(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());
```

> 注：后续我们会提供更方便的查阅 C++ API 文档的入口。

在 2.3 版本，我们共支持了大约 250 个类似的 C++ API，能够覆盖大部分的基础运算，但是除前述的 133 个 C++ API 之外，剩余的 C++ API 由于一些历史原因，其参数列表尚未和相应的 Python API 对齐，因此目前剩余这些 API 只能作为 experimental 的 API 使用，需要通过 `paddle::experimental::xxx` 进行调用，且这些 experimental API 在下个版本可能会有不兼容的升级，如果不介意随下一版本升级的话，可以使用，追求稳定的话则不建议使用。

如有需要，目前支持的全量 API 列表（包含 experimental API）请参考 paddle 安装路径下的 api.h 头文件，以 Python3.7 为例，其路径是 `python3.7/site-packages/paddle/include/paddle/phi/api/include/api.h`。


### 运算函数实现

对函数写法以及基础 API 的定义有了初步认识后，下面结合具体的示例进行介绍。

#### CPU 实现

以 `relu` 算子为例，一个支持 `float32` 类型的 CPU `relu` 算子运算函数可以实现如下：

- relu_cpu_fp32.cc

```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::empty_like(x);

  auto x_numel = x.numel();
  auto* x_data = x.data<float>();
  auto* out_data = out.data<float>();

  for (int64_t i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<float>(0.), x_data[i]);
  }

  return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::empty_like(x);

  auto out_numel = out.numel();
  auto* out_data = out.data<float>();
  auto* grad_out_data = grad_out.data<float>();
  auto* grad_x_data = grad_x.data<float>();

  for (int64_t i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<float>(0) ? 1. : 0.);
  }

  return {grad_x};
}
```

主要逻辑包括：

1. 创建输出 `Tensor`
2. 获取输入和输出 `Tensor` 的数据区起始地址
3. 计算得到输出 `Tensor` 的数值，返回结果

前述 `relu` 示例实现仅支持 `float32` 类型的计算，如果仅有一种数据类型的支持需求，用以上写法即可。

如果需要同时支持多种数据类型，例如同时支持 `float32` 与 `float64` 的计算，可以使用相应的 DIAPATCH 宏进行声明，示例如下：

- relu_cpu.cc

```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int64_t i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward_kernel", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.data<data_t>(), x.numel());
      }));

  return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.data<data_t>(),
                                   out.numel());
                             }));

  return {grad_x};
}
```

> 注：编写模板计算函数时，模板参数名 `data_t` 用于适配不同的数据类型，不可更改为其他命名，否则会编译失败

示例中的 `PD_DISPATCH_FLOATING_TYPES` 会展开得到 `float32` 与 `float64` 的 switch-case 实现，从而在运行时根据输入的数据类型，选择实际需要执行的分支。

例如，`ReluCPUForward` 中的 `PD_DISPATCH_FLOATING_TYPES` 实际代码展开如下：

```c++
switch(x.type()) {
  case paddle::DataType::FLOAT32:
    relu_cpu_forward_kernel<float>(
            x.data<float>(), out.data<float>(), x.numel());
    break;
  case paddle::DataType::FLOAT64:
    relu_cpu_forward_kernel<double>(
            x.data<double>(), out.data<float>(), x.numel());
    break;
  default:
    PD_THROW(
      "function relu_cpu_forward_kernel is not implemented for data type `",
      paddle::ToString(x.type()), "`");
}
```

目前定义的 dispatch 宏包括：

- `PD_DISPATCH_FLOATING_TYPES` ：dispatch 生成 `float` 和 `double` 对应的实现
- `PD_DISPATCH_FLOATING_AND_HALF_TYPES` ：dispatch 生成 `float` , `double` 和 `paddle::float16` 对应的实现
- `PD_DISPATCH_INTEGRAL_TYPES` ：dispatch 生成 `int8_t`, `uint8_t`, `int16_t`, `int`的`int64_t` 对应的实现
- `PD_DISPATCH_COMPLEX_TYPES`：dispatch 生成 `paddle::complex64` 和 `paddle::complex128` 对应的实现
- `PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES` ：dispatch 生成前述 `PD_DISPATCH_FLOATING_TYPES` 和 `PD_DISPATCH_INTEGRAL_TYPES` 两个宏全部数据类型对应的实现
- `PD_DISPATCH_FLOATING_AND_COMPLEX_TYPES`：dispatch 生成前述 `PD_DISPATCH_FLOATING_TYPES` 和 `PD_DISPATCH_COMPLEX_TYPES` 两个宏全部数据类型对应的实现
- `PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES`：dispatch 生成前述 `PD_DISPATCH_FLOATING_TYPES` , `PD_DISPATCH_INTEGRAL_TYPES` 和 `PD_DISPATCH_COMPLEX_TYPES` 三个宏全部数据类型对应的实现

当然，如果这几个宏无法满足您实际使用的需求，您可以直接通过 `switch-case` 语句实现，将来视需求我们也会添加更多的宏。

#### CPU&CUDA 混合实现

通常只有 CPU 的算子实现是不够的，实际生产环境中一般需要使用 GPU 算子。此处将前述 `relu_cpu.cc` 中算子的 CPU 实现改为 GPU 示例如下：

- relu_cuda.cu
```c++
#include "paddle/extension.h"

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t* dy,
                                          const data_t* y,
                                          data_t* dx,
                                          int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  auto out = paddle::empty_like(x);

  int64_t numel = x.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out.data<data_t>(), numel);
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out) {
  auto grad_x = paddle::empty_like(x);

  int64_t numel = out.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x.data<data_t>(),
            numel);
      }));

  return {grad_x};
}
```

- relu_cuda.cc
```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);

std::vector<paddle::Tensor> ReluCUDAForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  return relu_cuda_forward(x);
}

std::vector<paddle::Tensor> ReluCUDABackward(const paddle::Tensor& x,
                                             const paddle::Tensor& out,
                                             const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  return relu_cuda_backward(x, out, grad_out);
}
```

在 `.cu` 文件中实现对应的 CUDA kernel 和计算函数，在 `.cc` 文件中声明调用即可。

注意这里的 `CHECK_INPUT` 也改为检查输入 `Tensor` 是否在 GPU 上，如果后续仍然在 CPU 上执行，将会报错如下，可以看到报错提示与 `CHECK_INPUT` 缩写提示一致。至于错误类型，`PaddlePaddle` 将外部扩展自定义算子视为第三方模块，错误类型统一为 `OSError: (External)` ，与其他第三方库报错类型一致。报错示例如下：

```
Traceback (most recent call last):
  File "relu_test_jit_dy.py", line 70, in <module>
    out = net(image)
  File "/usr/local/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 902, in __call__
    outputs = self.forward(*inputs, **kwargs)
  File "relu_test_jit_dy.py", line 45, in forward
    tmp_out = custom_ops.custom_relu(tmp1)
  File "/root/.cache/paddle_extensions/custom_jit_ops/custom_jit_ops.py", line 16, in custom_relu
    helper.append_op(type="custom_relu", inputs=ins, outputs=outs, attrs=attrs)
  File "/usr/local/lib/python3.7/site-packages/paddle/fluid/layer_helper.py", line 43, in append_op
    return self.main_program.current_block().append_op(*args, **kwargs)
  File "/usr/local/lib/python3.7/site-packages/paddle/fluid/framework.py", line 3079, in append_op
    kwargs.get("stop_gradient", False))
  File "/usr/local/lib/python3.7/site-packages/paddle/fluid/dygraph/tracer.py", line 45, in trace_op
    not stop_gradient)
OSError: (External) x must be a GPU Tensor.
  [/work/scripts/custom_op/guide/relu_cuda.cc:13] (at /work/paddle/paddle/fluid/framework/custom_operator.cc:168)
  [operator < custom_relu > error]
```

实际使用时，一般您只需要根据您实际使用的设备，编写对应设备的算子实现即可，例如您使用 GPU 训练，仅需要实现算子的 CUDA 版本即可使用，如果您需要您的自定义算子同时支持多种设备，例如同时支持 CPU 与 GPU，只需要将 CPU 和 GPU 的实现整合到一起，并在前反向函数中实现对应的分支即可，示例如下：

- relu.cc
```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_CPU_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int64_t i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cpu_forward(const paddle::Tensor& x) {
  CHECK_CPU_INPUT(x);

  auto out = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward_kernel", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.data<data_t>(), x.numel());
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cpu_backward(const paddle::Tensor& x,
                                              const paddle::Tensor& out,
                                              const paddle::Tensor& grad_out) {
  CHECK_CPU_INPUT(x);
  CHECK_CPU_INPUT(out);
  CHECK_CPU_INPUT(grad_out);

  auto grad_x = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.data<data_t>(),
                                   out.numel());
                             }));

  return {grad_x};
}

// NOTE: If your custom operator may be compiled in an environment with CUDA,
// or it may be compiled in an environment without CUDA, in order to adapt the
// compilation environment, you can use the PADDLE_WITH_CUDA macro control
// the CUDA related code.
#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);
#endif

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  if (x.is_cpu()) {
    return relu_cpu_forward(x);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return relu_cuda_forward(x);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom relu operator.");
  }
}

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  if (x.is_cpu()) {
    return relu_cpu_backward(x, out, grad_out);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return relu_cuda_backward(x, out, grad_out);
#endif
  } else {
    PD_THROW("Unsupported device type for backward function of custom relu operator.");
  }
}
```

- relu.cu
```c++
#include "paddle/extension.h"

#define CHECK_CUDA_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t* dy,
                                          const data_t* y,
                                          data_t* dx,
                                          int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  CHECK_CUDA_INPUT(x);

  auto out = paddle::empty_like(x);

  int64_t numel = x.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out.data<data_t>(), numel);
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out) {
  CHECK_CUDA_INPUT(x);
  CHECK_CUDA_INPUT(out);
  CHECK_CUDA_INPUT(grad_out);

  auto grad_x = paddle::empty_like(x);

  int64_t numel = out.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x.data<data_t>(),
            numel);
      }));

  return {grad_x};
}
```


### 维度与类型推导函数实现

`PaddlePaddle` 框架同时支持动态图与静态图的执行模式，在静态图模式下，组网阶段需要完成 `Tensor shape` 和 `dtype` 的推导，从而生成正确的模型描述，用于后续 Graph 优化与执行。因此，除了算子的运算函数之外，还需要实现前向运算的维度和类型的推导函数。

维度推导（InferShape）和类型推导（InferDtype）的函数写法也是有要求的，形式如下：

```c++
std::vector<std::vector<int64_t>> OpInferShape(std::vector<int64_t> x_shape, ...) {
  return {x_shape, ...};
}

std::vector<paddle::DataType> OpInferDtype(paddle::DataType x_dtype, ...) {
  return {x_dtype, ...};
}
```

函数的输入参数与返回值类型固定，具体类型如上述代码片段所示，其他要求如下：

- 函数输入参数与前述运算函数的输入 `Tensor` 按顺序一一对应，依次为输入参数的 `shape` 和 `dtype`，这里的对应规则为：
    - `paddle::Tensor` -> `std::vector<int64_t>`
    - `std::vector<paddle::Tensor>` -> `std::vector<std::vector<int64_t>>`
- 函数返回值 vector 中的 `shape` 或 `dtype` 信息也需要与返回 `Tensor` 按顺序一一对应
- 维度推导函数支持 `Attribute` 的输入，在实现维度推导函数时，可以不使用 `Attribute` 的输入参数，也可以使用，但如果要使用的话，需要和 Forward 函数的 `Attribute` 参数保持一致
- 类型推导函数不支持 `Attribute` 的输入

以 `relu` 为例，其维度与类型推导函数如下：

- relu_cpu_fp32.cc / relu_cpu.cc / relu_cuda.cc / relu.cc （需将以下代码追加到前述文件中）

```c++
// 维度推导
std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

// 类型推导
std::vector<paddle::DataType> ReluInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}
```

> 注：如果是 CUDA 算子，ReluInferShape 和 ReluInferDtype 仅需要在.cc 文件中实现，不需要在.cu 中重复实现

对于仅有一个输入 `Tensor` 和一个输出 `Tensor` 的自定义算子，如果输出 `Tensor` 和输入 `Tensor` 的 `shape` 和 `dtype` 一致，可以省略 `InferShape` 和 `InferDtype` 函数的实现，其他场景下均需要实现这两个函数。因此，对于这里的 `relu` 算子来说，这两个函数可以不写。

此外，以 `concat` 为例，如果其将 `axis` 参数作为前向函数的 `Attribute` 输入，其维度与类型推导函数如下：

```c++
// 前向函数
std::vector<paddle::Tensor> ConcatForwardStaticAxis(
    const std::vector<paddle::Tensor>& inputs, int64_t axis) { ... }

// 维度推导
std::vector<std::vector<int64_t>> ConcatInferShapeStaticAxis(
    const std::vector<std::vector<int64_t>>& input_shapes,
    int64_t axis) { ... }

// 类型推导
std::vector<paddle::DataType> ConcatInferDtypeStaticAxis(
    const std::vector<paddle::DataType>& input_dtypes) { ... }
```

### 构建算子

最后，需要调用 `PD_BUILD_OP` 系列宏，构建算子的描述信息，并关联前述算子运算函数和维度、类型推导函数。

我们提供了 3 个构建算子的宏：

- `PD_BUILD_OP` ：用于构建前向算子
- `PD_BUILD_GRAD_OP` ：用于构建前向算子对应的反向算子
- `PD_BUILD_DOUBLE_GRAD_OP` ：用于构建前反向算子对应的二阶反向算子

> 注：二阶以上的反向算子构建暂不支持。

对于 `relu` CPU 示例来说，构建算子描述如下：

- relu_cpu_fp32.cc / relu_cpu.cc （需将以下代码追加到前述文件中）

```c++
PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluCPUForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReluInferDtype));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluCPUBackward));
```

这里写法上需要注意以下几点：
- `PD_BUILD_OP` 系列宏后面的括号内为算子名，也是后面在 python 端使用的接口名，注意前后不需要引号，注意该算子名不能与 `PaddlePaddle` 内已有算子名重名，比如 `relu` 为 `PaddlePaddle` 内已有算子，如果直接使用 relu 作为算子名将无法注册成功，所以此处增加了前缀 `custom_`
- `PD_BUILD_OP`、 `PD_BUILD_GRAD_OP` 和 `PD_BUILD_DOUBLE_GRAD_OP` 构建同一个算子的前向、反向、二阶反向实现，宏后面使用的算子名需要保持一致，比如该示例中均使用 `custom_relu`
- `PD_BUILD_OP`、 `PD_BUILD_GRAD_OP` 和 `PD_BUILD_DOUBLE_GRAD_OP` 必须顺次调用，不允许在未调用 `PD_BUILD_OP` 构建前向算子的情况下，直接调用 `PD_BUILD_GRAD_OP` 构建反向算子
- Inputs 与 Outputs 的输入参数为 `std::vector<std::string>` ，依次是前面算子运算函数的输入输出 `Tensor` 的 name，需要按顺序一一对应，此处的 name 与函数输入参数的变量名没有强关联，比如函数输入参数是 `const paddle::Tensor& x` ，Inputs 中的 name 可以是 `Input, x, X, In` 等等
- `PD_BUILD_OP` 与 `PD_BUILD_GRAD_OP` 中的 Inputs 与 Outputs 的 name 有强关联，对于前向算子的某个输入，如果反向算子仍然要复用，那么其 name 一定要保持一致，因为内部执行时，会以 name 作为 key 去查找对应的变量，比如这里前向算子的 `X, Out` 与反向算子的 `X, Out` 指代同一个 `Tensor`
- 在声明反向算子的 Inputs 与 Outputs 时，前向 `Tensor` 对应的梯度 `Tensor` 名需要由 `paddle::Grad` 处理前向 `Tensor` 名得到，不能够随意声明，例如这里 `"X"` 对应的梯度 `Tensor` 名为 `paddle::Grad("X")`
- 如果算子的 Inputs 与 Outputs 中包含变长的 `Tensor` 输入和输出，其 `Tensor` 名需要由 `paddle::Vec` 方法处理得到，例如对于前述 `concat` 算子的前向输入 `const std::vector<paddle::Tensor>& inputs` ，其 `Tensor` 名可以为 `paddle::Vec("X")` ，对应的梯度 `Tensor` 名为 `paddle::Grad(paddle::Vec("X"))` ，此处 `paddle::Grad` 需要在 `paddle::Vec` 的外面
- 此处 `SetKernelFn` 、`SetInferShapeFn` 与 `SetInferDtypeFn` 中的 `PD_KERNEL` 、`PD_INFER_SHAPE` 、`PD_INFER_DTYPE` 宏用于自动转换并统一函数的签名，不可以省略
- 反向算子构建暂时不支持调用 `SetInferShapeFn` 和 `SetInferDtypeFn` 自定义维度与类型推导函数，框架会根据前向 `Tensor` 的 `shape` 和 `dtype` ，设定其对应梯度 `Tensor` 的 `shape` 和 `dtype`

如前述介绍，此处 `relu` 也可以省略 `InferShape` 和 `InferDtype` 函数的实现，因此也可以写为：

```c++
PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluCPUForward));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluCPUBackward));
```

类似地，GPU 示例构建算子描述如下，替换 `KernelFn` 即可：

- relu_cuda.cc （需将以下代码追加到前述文件中）

```c++
PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluCUDAForward));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluCUDABackward));
```

对于 `concat` 算子，其包含变长的输入输出，因此 `PD_BUILD_OP` 声明时需要用到 `paddle::Vec` 方法，示例如下：

```c++
PD_BUILD_OP(custom_concat_with_attr)
    .Inputs({paddle::Vec("X")})
    .Outputs({"Out"})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(ConcatForwardStaticAxis))
    .SetInferShapeFn(PD_INFER_SHAPE(ConcatInferShapeStaticAxis))
    .SetInferDtypeFn(PD_INFER_DTYPE(ConcatInferDtypeStaticAxis));

PD_BUILD_GRAD_OP(custom_concat_with_attr)
    .Inputs({paddle::Vec("X"), paddle::Grad("Out")})
    .Outputs({paddle::Grad(paddle::Vec("X"))})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(ConcatBackwardStaticAxis));
```

#### Attribute 声明

对于 `Attribute` 的声明，和 Inputs、Outputs 的声明有所不同，需要按照如下格式声明字符串：

`<name>: <attr-type-expr>`

其中，`name` 为 `Attribute` 变量的 name，`<attr-type-expr>` 为 `Attribute` 变量的类型，类型字符串需要与 C++类型严格一致。通过如下示例说明：

假如有前向运算函数形式如下：

```c++
std::vector<paddle::Tensor> AttrTestForward(
    const paddle::Tensor& x,
    bool bool_attr,
    int int_attr,
    float float_attr,
    int64_t int64_attr,
    const std::string& str_attr,
    const std::vector<int>& int_vec_attr,
    const std::vector<float>& float_vec_attr,
    const std::vector<int64_t>& int64_vec_attr,
    const std::vector<std::string>& str_vec_attr) {...}
```

对应的 BUILD_OP 写法为：

```c++
PD_BUILD_OP(attr_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"bool_attr: bool",
            "int_attr: int",
            "float_attr: float",
            "int64_attr: int64_t",
            "str_attr: std::string",
            "int_vec_attr: std::vector<int>",
            "float_vec_attr: std::vector<float>",
            "int64_vec_attr: std::vector<int64_t>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestForward));
```

如果该算子需要反向实现，反向算子的 `Attribute` 输入参数需要是前向算子 `Attribute` 输入参数的子集，不能新增前向算子没有的 `Attribute` ，示例如下：

```c++
std::vector<paddle::Tensor> AttrTestBackward(
    const paddle::Tensor& grad_out,
    int int_attr,
    const std::vector<float>& float_vec_attr,
    const std::vector<std::string>& str_vec_attr) {...}

PD_BUILD_GRAD_OP(attr_test)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"int_attr: int",
            "float_vec_attr: std::vector<float>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestBackward));
```

这里的 `int_attr`、`float_vec_attr`、`str_vec_attr` 均是前向算子声明中出现过的参数，这里仅限定 `Attrs` 方法中字符串的命名，函数的输入参数命名没有限制，只需要确保数据类型一致即可，例如这里 `AttrTestBackward` 也可以改为如下写法：

```c++
std::vector<paddle::Tensor> AttrTestBackward(
    const paddle::Tensor& grad_out,
    int a,
    const std::vector<float>& b,
    const std::vector<std::string>& c) {...}
```

### 其他功能

#### 支持自定义设备

首先请参考 [新硬件接入示例](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/custom_device_docs/custom_device_example_cn.html) 确保自定义设备已经注册完成。

如果 CPU 实现和 GPU 实现无法满足新硬件的需求，可以通过组合 C++ 运算 API 的方式，实现自定义算子。将前述 `relu_cpu.cc` 中的 CPU 实现改为组合 C++ 运算 API 的示例如下：

```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_CUSTOM_INPUT(x) PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

std::vector<paddle::Tensor> relu_custom_forward(const paddle::Tensor& x) {
  CHECK_CUSTOM_INPUT(x);
  auto out = paddle::relu(x);
  return {out};
}

std::vector<paddle::Tensor> relu_custom_backward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  CHECK_CUSTOM_INPUT(x);
  CHECK_CUSTOM_INPUT(out);
  auto grad_x = paddle::empty_like(x, x.dtype(), x.place());
  auto ones = paddle::experimental::full_like(x, 1.0, x.dtype(), x.place());
  auto zeros = paddle::experimental::full_like(x, 0.0, x.dtype(), x.place());
  auto condition = paddle::experimental::greater_than(x, zeros);

  grad_x = paddle::multiply(grad_out, paddle::where(condition, ones, zeros));

  return {grad_x};
}

std::vector<paddle::Tensor> relu_custom_double_backward(
    const paddle::Tensor& out, const paddle::Tensor& ddx) {
  CHECK_CUSTOM_INPUT(out);
  auto ddout = paddle::empty(out.shape(), out.dtype(), out.place());
  auto ones = paddle::experimental::full_like(out, 1.0, out.dtype(), out.place());
  auto zeros = paddle::experimental::full_like(out, 0.0, out.dtype(), out.place());
  auto condition = paddle::experimental::greater_than(out, zeros);

  ddout = paddle::multiply(ddx, paddle::where(condition, ones, zeros));

  return {ddout};
}

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  if (x.is_cpu()) {
    return relu_cpu_forward(x);
  } else if (x.is_custom_device()) {
    return relu_custom_forward(x);
  } else {
    PD_THROW("Not implemented.");
  }
}

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  if (x.is_cpu()) {
    return relu_cpu_backward(x, out, grad_out);
  } else if (x.is_custom_device()) {
    return relu_custom_backward(x, out, grad_out);
  } else {
    PD_THROW("Not implemented.");
  }
}

std::vector<paddle::Tensor> ReluDoubleBackward(const paddle::Tensor& out,
                                               const paddle::Tensor& ddx) {
  if (out.is_cpu()) {
    return relu_cpu_double_backward(out, ddx);
  } else if (out.is_custom_device()) {
    return relu_custom_double_backward(out, ddx);
  } else {
    PD_THROW("Not implemented.");
  }
}
```

支持的 C++ 运算 API 可参考 [类 Python 的 C++运算 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/custom_op/new_cpp_op_cn.html#python-c-api)

##### 获取自定义设备的 stream

用户想要获取设备的 `stream` 时，可以通过下述方式获取对应 `Tensor` 的 `stream`（需要添加头文件 `#include "paddle/phi/backends/all_context.h"`，当前方法尚不稳定，在下个版本有不兼容升级的可能，如果不介意随下一版本升级的话，可以使用，追求稳定的话则不建议使用）：

```c++
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

void* GetStream(const paddle::Tensor& x) {
  CHECK_CUSTOM_INPUT(x);

  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  void* stream = custom_ctx->stream();
  PD_CHECK(stream != nullptr);

  return stream;
}
```

#### inplace 机制

使用 inplace 机制定义的自定义算子，可以指定输入和输出使用同一个 Tensor，或者对输入的 Tensor 做原位修改。

下面结合具体的使用示例进行介绍，将 `relu` 算子改写为 inplace 算子，函数实现如下：

```c++
#include "paddle/extension.h"

#include <vector>

template <typename data_t>
void relu_forward_kernel(data_t* x_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    x_data[i] = x_data[i] > 0 ? x_data[i] : 0;
  }
}

template <typename data_t>
void relu_backward_kernel(const data_t* out_data,
                          data_t* grad_out_data,
                          int64_t out_numel) {
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_out_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

void ReluCpuInplaceForward(paddle::Tensor& x) {  // NOLINT
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");

  PD_DISPATCH_FLOATING_TYPES(x.type(), "ReluForward", ([&] {
                               relu_forward_kernel<data_t>(x.data<data_t>(),
                                                           x.size());
                             }));
}

void ReluCpuInplaceBackward(const paddle::Tensor& x,
                            const paddle::Tensor& out,
                            paddle::Tensor& grad_out) {  // NOLINT
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");

  PD_DISPATCH_FLOATING_TYPES(
      grad_out.type(), "ReluBackward", ([&] {
        relu_backward_kernel<data_t>(
            out.data<data_t>(), grad_out.data<data_t>(), grad_out.size());
      }));
}

PD_BUILD_OP(custom_inplace_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetInplaceMap({{"X", "Out"}})
    .SetKernelFn(PD_KERNEL(ReluCpuInplaceForward));

PD_BUILD_GRAD_OP(custom_inplace_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetInplaceMap({{paddle::Grad("Out"), paddle::Grad("X")}})
    .SetKernelFn(PD_KERNEL(ReluCpuInplaceBackward));
```

相比于 `relu` 算子的常规实现，使用 inplace 机制需要注意以下几点：

1. 输入的 inplace Tensor 类型，应该修改为 `paddle::Tensor&` 而非 `const paddle::Tensor&`；

2. 定义算子时，需要使用 `SetInplaceMap` 指明输入和输出间 inplace 的映射关系。`SetInplaceMap` 传入的参数类型为 `std::unordered_map<std::string, std::string>`，支持多组输入和输出之间进行 inplace 映射。例如可以定义： `.SetInplaceMap({{"X", "Out1"}, {"Y", "Out2"}})`；

3. 一方面，做 inplace 映射的输出 Tensor，不再作为函数的返回值，如果此时函数没有需要返回的 Tensor，函数的输出类型应为 `void` ；另一方面，其他没有做 inplace 映射的输出 Tensor，仍需作为返回值显式输出，此时函数的输出类型仍为 `std::vector<paddle::Tensor>`。例如 `ReluCpuInplaceForward` 函数中不再显式输出 Tensor，因此函数返回类型为 `void`；

4. 框架会自动为 inplace 的输入输出做 Shape 和 Dtype 映射。因此 `InferShape` 和 `InferDtype` 函数只需要返回未被 inplace 映射的输出类型。如果没有需要返回的值，可以不设置这两个函数。

5. 框架会对算子的输入、输出映射做基本的正确性检查（`SetInplaceMap`中指定的输入 Tensor 命名与 `Inputs` 中定义的名称一致；输出 Tensor 命名与 `Outputs` 中定义的名称一致），因此 `SetInplaceMap` 必须在 `Inputs` 和 `Outputs` 之后指定。

下面以一个自定义的 inplace `custom_add` 加法实现为例，来对上述的注意事项进行介绍：


```c++
#include "paddle/extension.h"

#include <vector>

template <typename data_t>
void add_forward_kernel(data_t* x_data, const data_t* y_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    x_data[i] += y_data[i];
  }
}

template <typename data_t>
void add_backward_kernel(data_t* y_grad_data,
                         const data_t* out_grad_data,
                         int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    y_grad_data[i] = out_grad_data[i];
  }
}

// 有 inplace 映射的输出 Tensor，不再作为函数的返回值，如果此时函数没有需要返回的 Tensor，函数的输出类型应为 `void`
void AddForward(paddle::Tensor& x,          // 输入的 inplace Tensor 类型，应该修改为 `paddle::Tensor&`
                const paddle::Tensor& y) {
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");

  PD_DISPATCH_FLOATING_TYPES(x.type(), "AddForward", ([&] {
                               add_forward_kernel<data_t>(x.data<data_t>(),
                                                          y.data<data_t>(),
                                                          x.size());
                             }));
  // 输出 Tensor out 指定了 inplace 映射，因此不需要显式的返回
}

// 输入的 Tensor 已通过 inplace 指定，不需要设置 InferShapeFn 和 InferDtypeFn

// 没有做 inplace 映射的输出 Tensor，仍需作为返回值显式输出，此时函数的输出类型仍为 std::vector<paddle::Tensor>
std::vector<paddle::Tensor> AddBackward(const paddle::Tensor& x,
                                        const paddle::Tensor& y,
                                        paddle::Tensor& out_grad) {  // 输入的 inplace Tensor 类型，应该修改为 `paddle::Tensor&`
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");
  PD_CHECK(y.place() == paddle::PlaceType::kCPU, "y must be a CPU Tensor.");

  paddle::Tensor y_grad = paddle::empty(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      out_grad.type(), "AddBackward", ([&] {
        add_backward_kernel<data_t>(
            y_grad.data<data_t>(), out_grad.data<data_t>(), out_grad.size());
      }));

  // y_grad 没有指定 inplace 映射，因此仍然需要显式的作为返回值
  return {y_grad};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetInplaceMap({{"X", "Out"}})                  // 使用 `SetInplaceMap` 指明输入和输出间 inplace 的映射关系
    .SetKernelFn(PD_KERNEL(AddForward));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", "Y", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .SetInplaceMap({{paddle::Grad("Out"), paddle::Grad("X")}})  // `SetInplaceMap` 必须在 `Inputs` 和 `Outputs` 之后指定
    .SetKernelFn(PD_KERNEL(AddBackward));

```


#### optional 机制

自定义算子的 optional 机制主要用于传入 Tensor 可能为 None 的场景，C++ 算子通过判断输入的 optional Tensor 是否为 None，可以执行不同的操作。

下面结合具体的使用示例进行介绍，自定义一个输入为 `Tensor x` 和 `optional<Tensor> y`，输出为 `Tensor out` 的加法算子：

$$
out =
\begin{cases}
x + y, & \text{   if  } y \text{  is valid}\\\\
x + x, & \text{   if  } y \text{  is none}
\end{cases}
$$

函数实现如下：
```c++
#include <vector>

#include "paddle/extension.h"

/*
if (y) {
  out = x + y;
} else {
  out = x + x;
}
*/
std::vector<paddle::Tensor> AddForward(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& y) {  // NOLINT
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");
  paddle::Tensor out = paddle::empty(x.shape(), x.dtype(), x.place());

  if (y) {
    out = x + y.get();
  } else {
    out = x + x;
  }

  return {out};
}

std::vector<paddle::DataType> AddInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<paddle::DataType>& y_dtype) {
  if (y_dtype) {
    return {*y_dtype};
  }
  return {x_dtype};
}

std::vector<std::vector<int64_t>> AddInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<int64_t>>& y_shape) {
  if (y_shape) {
    return {*y_shape};
  }
  return {x_shape};
}

/*
if (y) {
  x_grad = out_grad;
} else {
  x_grad = out_grad + out_grad;
}
*/
std::vector<paddle::Tensor> AddBackward(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& y,
    const paddle::Tensor& out_grad) {  // NOLINT
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");
  paddle::Tensor x_grad = paddle::zeros(x.shape(), x.dtype(), x.place());

  if (y) {
    x_grad = out_grad;
  } else {
    x_grad = out_grad + out_grad;
  }

  return {x_grad};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", paddle::Optional("Y")})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddInferDtype));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", paddle::Optional("Y"), paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(AddBackward));
```

相比于算子的常规实现，使用 optional 机制需要注意以下几点：

1. 输入的 optional Tensor 类型，应该修改为 `const paddle::optional<paddle::Tensor>&` 而非 `const paddle::Tensor&`；相应的 `InferShapeFn` 和 `InferDtypeFn` 输入类型分别修改为 `const paddle::optional<std::vector<int64_t>>&` 和 `const paddle::optional<paddle::DataType>&`；

2. 定义算子时，需要使用 `paddle::Optional` 标注 optional 类型的 Tensor；

3. 暂不支持 optional\<Tensor\> 类型的输出，因此反向算子做计算时，无法输出前向算子 optional Tensor 类型输入的梯度。

4. optional 的定义可以参考源码文件 `paddle/utils/optional.h`，用法与 boost optional 基本一致。

## 自定义算子编译与使用

本机制提供了两种编译自定义算子的方式，分别为 **使用 `setuptools` 编译** 与 **即时编译** ，下面依次通过示例介绍。

> 注：在进行编译之前，需要根据实际需求，将前述 **运算函数实现** ，**维度与类型推导函数实现** ， **构建算子** 三节中的代码示例组合到一起，具体地，需要将 **维度与类型推导函数实现** ， **构建算子** 两节中的代码片段追加到  **运算函数实现** 小节中对应的 *.cc 文件中

### 使用 `setuptools` 编译

该方式是对 `python` 内建库中的 `setuptools.setup` 接口的进一步封装，能够自动地生成 Python API 并以 Module 的形式安装到 site-packages 目录。编译完成后，支持通过 import 语句导入使用。

您需要编写 `setup.py` 文件， 配置自定义算子的编译规则。

例如，前述 `relu` 示例的 `setup` 文件可以实现如下：

- setup_cpu.py ( for relu_cpu.cc )

```python
from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CppExtension(
        sources=['relu_cpu.cc']
    )
)
```

- setup_cuda.py ( for relu_cuda.cc & relu_cuda.cu )

```python
from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=['relu_cuda.cc', 'relu_cuda.cu']
    )
)
```

其中 `paddle.utils.cpp_extension.setup` 能够自动搜索和检查本地的 `cc(Linux)` 、 `cl.exe(Windows)` 和 `nvcc` 编译命令和版本环境，根据用户指定的 `Extension` 类型，完成 CPU 或 GPU 设备的算子编译安装。

执行 `python setup_cpu.py install` 或者 `python setup_cuda.py install` 即可一键完成自定义算子的编译和安装。

以 `python setup_cuda.py install` 为例，执行日志如下：

```
running install
running bdist_egg
running egg_info
writing custom_setup_ops.egg-info/PKG-INFO
writing dependency_links to custom_setup_ops.egg-info/dependency_links.txt
writing top-level names to custom_setup_ops.egg-info/top_level.txt
reading manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
installing library code to build/custom_setup_ops/bdist.linux-x86_64/egg
running install_lib
running build_ext
/usr/local/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return (isinstance(seq, collections.Sequence) and
Compiling user custom op, it will cost a few seconds.....
creating build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.7/version.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.7/relu_cpu.o -> build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.7/relu_cuda.o -> build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.7/relu_cuda.cu.o -> build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.7/custom_setup_ops.so -> build/custom_setup_ops/bdist.linux-x86_64/egg
creating stub loader for custom_setup_ops.so
byte-compiling build/custom_setup_ops/bdist.linux-x86_64/egg/custom_setup_ops.py to custom_setup_ops.cpython-37.pyc
creating build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/PKG-INFO -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/SOURCES.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/dependency_links.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/not-zip-safe -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/top_level.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
writing build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
creating 'dist/custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg' and adding 'build/custom_setup_ops/bdist.linux-x86_64/egg' to it
removing 'build/custom_setup_ops/bdist.linux-x86_64/egg' (and everything under it)
Processing custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg
creating /usr/local/lib/python3.7/site-packages/custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg
Extracting custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg to /usr/local/lib/python3.7/site-packages
Adding custom-setup-ops 0.0.0 to easy-install.pth file

Installed /usr/local/lib/python3.7/site-packages/custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for custom-setup-ops==0.0.0
Finished processing dependencies for custom-setup-ops==0.0.0
```

执行成功后，如日志所示，自定义算子模块 `custom_setup_ops` 被安装至如下目录：

`/usr/local/lib/python3.7/site-packages/custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg`

`custom_setup_ops-0.0.0-py3.7-linux-x86_64.egg` 目录中内容如下：

```
custom_setup_ops_pd_.so  EGG-INFO/     relu_cpu.o      relu_cuda.o
custom_setup_ops.py      __pycache__/  relu_cuda.cu.o  version.txt
```

其中 `custom_setup_ops_pd_.so` 为自定义算子编译生成的动态库， `custom_setup_ops.py` 为根据 `PaddlePaddle` 接口的定义规则，自动生成的自定义算子 python 模块源码，其示例内容为（自动生成的代码后续可能会更新，生成结果可能与示例代码不一致）：

```python
import paddle.fluid.core as core
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper

def custom_relu(x):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dygraph_mode():
        res = core.eager._run_custom_op("custom_relu", x)
        return res[0] if len(res)==1 else res
    else:
        ins = {'X' : x}
        outs = {}
        outs_list = ['Out']
        helper = LayerHelper("custom_relu", **locals())

        outs['Out'] = helper.create_variable(dtype='float32')
        helper.append_op(type="custom_relu", inputs=ins, outputs=outs, attrs={})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res)==1 else res


import os
import sys
import types
import paddle
import importlib.util

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "custom_relu_module_setup_pd_.so")

def __bootstrap__():
    assert os.path.exists(so_path)
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
    for custom_ops in custom_ops:
        setattr(mod, custom_ops, eval(custom_ops))

__bootstrap__()

```

随后，可以直接在构建模型过程中导入使用，简单示例如下：

```python
import paddle
from custom_setup_ops import custom_relu

x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_relu(x)
```

> 注：`setuptools` 的封装是为了简化自定义算子编译和使用流程，即使不依赖于 `setuptools` ，也可以自行编译生成动态库，并封装相应的 python API，然后在基于 `PaddlePaddle` 实现的模型中使用

如果需要详细了解相关接口，或需要配置其他编译选项，请参考以下 API 文档：

- [paddle.utils.cpp_extension.setup](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/setup_cn.html)
- [paddle.utils.cpp_extension.CppExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CppExtension_cn.html)
- [paddle.utils.cpp_extension.CUDAExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CUDAExtension_cn.html)

### 即时编译（`JIT Compile`）

即时编译将 `setuptools.setup` 编译方式做了进一步的封装，通过将自定义算子对应的 `.cc` 和 `.cu` 文件传入 API `paddle.utils.cpp_extension.load`，在后台生成 `setup.py` 文件，并通过子进程的方式，隐式地执行源码文件编译、符号链接、动态库生成、组网 API 接口生成等一系列过程。不需要本地预装 CMake 或者 Ninja 等工具命令，仅需必要的编译器命令环境。 Linux 下需安装版本不低于 5.4 的 GCC，并软链到 `/usr/bin/cc` ，Windows 下需安装版本不低于 2017 的 Visual Studio；若编译支持 GPU 设备的算子，则需要提前安装 CUDA，其中自带 `nvcc` 编译环境。

对于前述 `relu` 示例，使用方式如下：

- for relu_cuda.cc & relu_cuda.cu
```python
import paddle
from paddle.utils.cpp_extension import load

custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])

x = paddle.randn([4, 10], dtype='float32')
out = custom_ops.custom_relu(x)
```

`load` 返回一个包含自定义算子 API 的 `Module` 对象，可以直接使用自定义算子 name 调用 API。

以 Linux 平台为例，`load` 接口调用过程中，如果不指定 `build_directory` 参数，Linux 会默认在 `~/.cache/paddle_extensions` 目录下生成一个 `{name}_setup.py`（Windows 默认目录为 `C:\\Users\\xxx\\.cache\\paddle_extensions` 用户目录），然后通过 subprocess 执行 `python {name}_setup.py build`，然后载入动态库，生成 Python API 之后返回。

对于本示例，默认生成路径内容如下：

```
λ ls ~/.cache/paddle_extensions/
custom_jit_ops/  custom_jit_ops_setup.py
```

其中，`custom_jit_ops_setup.py` 是生成的 setup 编译文件，`custom_jit_ops` 目录是编译生成的内容。

如果需要详细了解 load 接口，或需要配置其他编译选项，请参考 API 文档 [paddle.utils.cpp_extension.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/load_cn.html) 。

### 同时编译多个算子

以上两种方式均支持同时编译多个自定义算子，只需要将多个算子对应的源文件均传入对应的参数，编译生成的动态库中会包含多个算子的实现，导入 Module 之后，同样以算子名作为 API 名进行调用，示例如下：

- setuptools 编译
```python
from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu']
    )
)
```

> 注：此处需要是多个不同算子的实现，而不能是同一个算子的不同版本实现，例如这里不能将前述的 `relu_cpu.cc` 和 `relu_cuda.cc/cu` 一起编译，因为他们的算子名是相同的，都是 `custom_relu` ， 如果需要同一个算子在不同设备上的实现，建议将不同设备上的实现整合到一起，例如前述的 `relu.cc/cu`

调用方式：

```python
import paddle
# Suppose the op names are `custom_relu` and `custom_tanh`
from custom_ops import custom_relu, custom_tanh

x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_relu(x)
tanh_out = custom_tanh(x)
```

- JIT compile
```python
from paddle.utils.cpp_extension import load

custom_ops = load(
    name='custom_jit_ops',
    sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu'])

x = paddle.randn([4, 10], dtype='float32')
# Suppose the op names are `custom_relu` and `custom_tanh`
relu_out = custom_ops.custom_relu(x)
tanh_out = custom_ops.custom_tanh(x)
```

### ABI 兼容性检查

以上两种方式，编译前均会执行 ABI 兼容性检查 。对于 Linux，会检查 cc 命令对应的 GCC 版本是否与所安装的 `PaddlePaddle` 的 GCC 版本一致。例如对于 CUDA 10.1 以上的 `PaddlePaddle` 默认使用 GCC 8.2 编译，则本地 cc 对应的编译器版本也需为 8.2。对于 Windows，则会检查本地的 Visual Studio 版本是否与所安装的 `PaddlePaddle` 的 Visual Studio 版本一致（>=2017）。如果上述版本不一致，则会打印出相应 warning，且可能引发自定义 OP 编译执行报错。

## 在模型中使用自定义算子

经过前述过程，自定义算子的编写、编译安装及 API 生成均已完成，现在您可以在网络模型中使用您自定义生成的算子了，本方案生成的自定义算子在动态图和静态图模式下均能够使用。

以下验证用例均基于前述源文件 `relu_cuda.cc` 和 `relu_cuda.cu` 测试 `custom_relu` 在 GPU 环境中的使用，均采用 JIT Compile 的方式编译自定义算子。

通过定义一个简单的网络模型，完成训练迭代和存储推理模型的基本过程。

### 动态图模式

动态图模式的使用示例如下：

```python
import numpy as np

import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
from paddle.utils.cpp_extension import load

EPOCH_NUM = 4
BATCH_SIZE = 64

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])


class LeNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = custom_ops.custom_relu(x)
        x = self.max_pool1(x)
        x = custom_ops.custom_relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        x = custom_ops.custom_relu(x)
        x = self.linear3(x)
        return x


# set device
paddle.set_device("gpu")

# model
net = LeNet()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# data loader
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(train_loader()):
        out = net(image)
        loss = loss_fn(out, label)
        loss.backward()

        if batch_id % 300 == 0:
            print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, paddle.mean(loss).numpy()))

        opt.step()
        opt.clear_grad()

# save inference model
path = "custom_relu_test_dynamic/net"
paddle.jit.save(net, path,
    input_spec=[paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32')])
```

### 静态图模式

静态图模式的使用示例如下：

```python
import numpy as np

import paddle
from paddle import nn
from paddle import static
from paddle.utils.cpp_extension import load
from paddle.vision.transforms import Compose, Normalize

EPOCH_NUM = 4
BATCH_SIZE = 64

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops", sources=["relu_cuda.cc", "relu_cuda.cu"]
)


class LeNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1
        )
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = custom_ops.custom_relu(x)
        x = self.max_pool1(x)
        x = custom_ops.custom_relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        x = custom_ops.custom_relu(x)
        x = self.linear3(x)
        return x


# set device
paddle.enable_static()
paddle.set_device("gpu")

# model
image = static.data(shape=[None, 1, 28, 28], name='image', dtype='float32')
label = static.data(shape=[None, 1], name='label', dtype='int64')

net = LeNet()
out = net(image)
loss = nn.functional.cross_entropy(out, label)

opt = paddle.optimizer.Adam(learning_rate=0.001)
opt.minimize(loss)

# data loader
transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

# prepare
exe = static.Executor()
exe.run(static.default_startup_program())

compiled_program = static.CompiledProgram(static.default_main_program())

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image_data, label_data) in enumerate(train_loader()):
        loss_data = exe.run(
            compiled_program,
            feed={'image': image_data, 'label': label_data},
            fetch_list=[loss],
        )
        if batch_id % 300 == 0:
            print(
                "Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss_data)
                )
            )

# save inference model
path = "custom_relu_test_static/net"
static.save_inference_model(path, [image], [out], exe)
```

## 算子在推理场景中的使用

基于本机制编写的自定义算子，也能够在 `PaddlePaddle` 推理场景中使用，仍然基于前述示例介绍使用流程，这里基于 `relu_cuda.cc` 和 `relu_cuda.cu` 介绍。

### 算子与推理库联合编译

编写推理的测试程序，其中需要使用前述验证过程中存储的 inference model，目录为 `custom_relu_dynamic/net` 或者 `custom_relu_static/net` ，下面通过示例介绍使用流程，该示例需要准备的文件包括：

```
- cmake
  - external
    - boost.cmake
- CMakeLists.txt
- custom_op_test.cc
- relu_cuda.cc
- relu_cuda.cu
- run.sh
```

下面依次对各新增文件进行介绍。

#### 编写推理程序

下面是一个简单的推理 Demo，导入前述 `custom_relu_dynamic/net` 中存储的模型和参数，进行预测：

```c++
#include <numeric>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

void run(Predictor *predictor, const std::vector<float> &input,
         const std::vector<int> &input_shape, std::vector<float> *out_data) {
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
}

int main() {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel("custom_relu_dynamic/net.pdmodel",
                  "custom_relu_dynamic/net.pdiparams");
  auto predictor{paddle_infer::CreatePredictor(config)};
  std::vector<int> input_shape = {1, 1, 28, 28};
  std::vector<float> input_data(1 * 1 * 28 * 28, 1);
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);
  for (auto e : out_data) {
    LOG(INFO) << e << '\n';
  }
  return 0;
}
```

#### 编写 CMake 文件

编写 `CMakeList` 编译构建文件，示例如下：

在当前目录创建文件 `CMakeLists.txt` ，其内容为：

- CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(USE_TENSORRT "Compile demo with TensorRT."   ON)
option(CUSTOM_OPERATOR_FILES "List of file names for custom operators" "")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WITH_GPU)
  find_package(CUDA REQUIRED)
  add_definitions("-DPADDLE_WITH_CUDA")
endif()

if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    if (WITH_MKL)
      set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    safe_set_static_flag()
    if (WITH_STATIC_LIB)
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()
else()
  if(WITH_MKL)
    set(FLAG_OPENMP "-fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ${FLAG_OPENMP}")
endif()

if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
  else()
    if(CUDA_LIB STREQUAL "")
      set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64")
    endif()
  endif(NOT WIN32)
endif()

if (USE_TENSORRT AND WITH_GPU)
  set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
  if("${TENSORRT_ROOT}" STREQUAL "")
      message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
  endif()
  set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
  set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
endif()

if (NOT WIN32)
  if (USE_TENSORRT AND WITH_GPU)
      include_directories("${TENSORRT_INCLUDE_DIR}")
      link_directories("${TENSORRT_LIB_DIR}")
  endif()
endif(NOT WIN32)

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  if(WIN32)
    set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    endif(WIN32)
  endif()
else()
  set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
  include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
  if(WIN32)
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if(WITH_STATIC_LIB)
  set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()

if (NOT WIN32)
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf  xxhash
      ${EXTERNAL_LIB})
else()
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags_static libprotobuf  xxhash ${EXTERNAL_LIB})
  set(DEPS ${DEPS} shlwapi.lib)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    if(USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
  endif()
endif()

cuda_add_library(pd_infer_custom_op ${CUSTOM_OPERATOR_FILES} SHARED)
add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)
set(DEPS ${DEPS} boost pd_infer_custom_op)

if(WIN32)
  if(USE_TENSORRT)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
              ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    )
  endif()
  if(WITH_MKL)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
          COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_PATH}/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release
    )
  else()
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${CMAKE_BINARY_DIR}/Release
    )
  endif()
  if(NOT WITH_STATIC_LIB)
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_LIB}/paddle/lib/paddle_fluid.dll" ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
      )
  endif()
endif()

target_link_libraries(${DEMO_NAME} ${DEPS})
```

#### 编写编译执行脚本

编写编译执行脚本 `run.sh` ，脚本内容如下：

- run.sh
```sh
mkdir -p build
cd build
rm -rf *

DEMO_NAME=custom_op_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir
CUDNN_LIB=/usr/local/cudnn/lib64
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/root/work/nvidia/TensorRT-6.0.1.5.cuda-10.1.cudnn7.6-OSS7.2.1
CUSTOM_OPERATOR_FILES="relu_cuda.cc;relu_cuda.cu"


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES}

make -j
```

此处要根据实际情况对执行脚本中的几处配置进行调整：

```sh
# 根据预编译库中的 version.txt 信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir

# 如果上述的 WITH_GPU 或 USE_TENSORRT 设为 ON，请设置对应的 CUDA， CUDNN， TENSORRT 的路径。
CUDNN_LIB=/paddle/nvidia-downloads/cudnn_v7.5_cuda10.1/lib64
CUDA_LIB=/paddle/nvidia-downloads/cuda-10.1/lib64
# TENSORRT_ROOT=/paddle/nvidia-downloads/TensorRT-6.0.1.5
```

然后，运行 `sh run.sh` ，完成编译，会在目录下产生 build 目录。

### 运行推理程序

```
# 进入 build 目录
cd build
# 运行样例
./custom_op_test
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

### 参考链接汇总

- [Paddle Inference 快速开始](https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html)
- [Paddle Inference API 文档](https://paddleinference.paddlepaddle.org.cn/api_reference/cxx_api_index.html)
- [更多示例代码-自定义算子单元测试](https://github.com/PaddlePaddle/Paddle/tree/develop/test/custom_op)

API 文档：
- [paddle.utils.cpp_extension.setup](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/setup_cn.html)
- [paddle.utils.cpp_extension.CppExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CppExtension_cn.html)
- [paddle.utils.cpp_extension.CUDAExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CUDAExtension_cn.html)
- [paddle.utils.cpp_extension.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/load_cn.html)
