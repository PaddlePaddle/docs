# 自定义外部算子（新）

## 概述

算子（Operator，简称Op）是构建神经网络的基础组件，飞桨框架提供了丰富的算子库，能够满足绝大多数场景的使用需求。但是出于以下几点原因，您可能希望定制化算子的C++实现，从而满足特定需求：

1. 已有的算子无法组合出您需要的运算逻辑；
2. 使用已有算子组合得到的运算逻辑无法满足您的性能需求。

为此，我们提供了自定义外部算子的机制，以此机制实现的自定义算子，能够以 **即插即用** 的方式用于模型训练与推理，不需要重新编译安装飞桨框架。

使用自定义算子机制，仅需要以下两个步骤：

1. 实现算子的C++运算逻辑
2. 调用相关接口完成算子编译与注册

随后即可在模型中使用，下面通过实现一个 `relu` 运算，介绍具体的实现、编译与应用流程。

> 注意事项：
> - 在使用本机制实现自定义算子之前，请确保已经正确安装了 `PaddlePaddle 2.0.1` 及以上版本
> - 该机制目前仅支持 `Linux` 与 `Windows` 平台，`Mac` 平台会在后续版本支持

## 算子C++实现

自定义算子需要实现编写以下组件的C++实现，包括：

1. 算子的运算函数：算子核心的计算逻辑实现，主要是对输入Tensor进行处理，得到输出Tensor的过程
2. 算子的维度与类型推导函数：用于在组网编译和运行时，正确推导出输出Tensor的shape和data type
3. 算子描述：描述算子的输入输出信息、并关联前述运算、维度推导与类型推导函数

下面结合示例进行介绍。

### 运算函数与基础API

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

- 函数输入参数可以是Tensor或者一些基础类型的Attribute，具体地：
    - Tensor必须以 `const paddle::Tensor& ` 的形式作为输入，可以有一个或多个
    - Attribute目前仅支持如下数据类型，可以有一个或多个：
        - `bool`
        - `int`
        - `float`
        - `int64_t`
        - `std::string`
        - `std::vector<int>`
        - `std::vector<float>`
        - `std::vector<int64_t>`
        - `std::vector<std::string>`
    - Attribute参数必须在所有Tensor参数之后
- 函数返回值只能是 `std::vector<paddle::Tensor>`

> 注：其他类型的数值作为函数输入参数或者返回值将无法编译通过

对于基础的设备和数据类型支持情况，我们定义了两个简单的枚举类：

- 设备表示：`enum class PlaceType { kUNK = -1, kCPU, kGPU };`
- 数据类型表示：`enum class DataType {BOOL, INT8, UINT8, INT16, INT32, INT64, FLOAT32, FLOAT64};`

> 注：目前仅支持以上设备与数据类型，其他类型会在后续版本支持

对于 `paddle::Tensor` ，我们目前提供了一些基础的API，包括：

- 构造API：
    - `Tensor(const PlaceType& place)`：输入参数place，返回一个Tensor对象
- 设备相关API：
    - `const PlaceType& place() const`：获取Tensor所在的设备
- 数据类型相关API：
    - `DataType type() const`：获取Tensor的数据类型
- 长度与维度相关API：
    - `int64_t size() const`：获取Tensor的数据长度
    - `std::vector<int64_t> shape() const`：获取Tensor的维度信息
    - `void reshape(const std::vector<int64_t>& shape)`：输入参数shape，修改Tensor的维度信息
- 数据访问API：
    - `template <typename T> T* data() const`：
        - 模板类方法，获取数据内存的起始地址（只读访问）
    - `template <typename T> T* mutable_data(const PlaceType& place)`：
        - 模板类方法，输入参数place，根据Tensor.shape在指定设备上申请内存，并返回内存的起始地址
- 工具类API：
    - `template <typename T> Tensor copy_to(const PlaceType& place) const`：
        - 模板类方法，输入参数place，将当前Tensor拷贝到指定设备上并返回
    - `Tensor cast(const DataType& target_type) const`：
        - 输入参数target_type，将当前Tensor转换为指定数据类型的Tensor并返回
    - `cudaStream_t stream() const`：
        - 用于获取当前Tensor所处的CUDA Stream（仅在GPU编译版本中生效）
        - 仅能够获取函数输入Tensor的stream

> 注：后续会继续扩展其他API，API的声明详见 `paddle/paddle/fluid/extension/include/` 目录下的头文件

对函数签名以及基础API的定义有了初步认识后，下面结合具体的示例进行介绍。

#### CPU实现

以 `relu` 算子为例，一个支持 `float32` 类型的CPU `relu` 算子运算函数可以实现如下：

- relu_cpu_fp32.cc

```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU);
  out.reshape(x.shape());

  auto x_numel = x.size();
  auto* x_data = x.data<float>();
  auto* out_data = out.mutable_data<float>(x.place());

  for (int i = 0; i < x_numel; ++i) {
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

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU);
  grad_x.reshape(x.shape());

  auto out_numel = out.size();
  auto* out_data = out.data<float>();
  auto* grad_out_data = grad_out.data<float>();
  auto* grad_x_data = grad_x.mutable_data<float>(x.place());

  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<float>(0) ? 1. : 0.);
  }

  return {grad_x};
}
```

主要逻辑包括：

1. 创建输出的Tensor，设定其shape
2. 获取输入Tensor的数据区起始地址，为输出Tensor申请内存并返回数据区起始地址
3. 计算得到输出Tensor的数值，返回结果

> 注：目前尚不支持输入Tensor的inplace改动，将会在后续版本支持

前述 `relu` 示例实现仅支持 `float32` 类型的计算，如果仅有一种数据类型的支持需求，用以上写法即可。

如果需要同时支持多种数据类型，例如同时支持 `float32` 与 `float64` 的计算，可以使用相应的dispatch宏进行声明，示例如下：

- relu_cpu.cc

```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU);
  out.reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward_kernel", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU);
  grad_x.reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(x.place()),
                                   out.size());
                             }));

  return {grad_x};
}
```

> 注：编写模板计算函数时，模板参数名 `data_t` 用于适配不同的数据类型，不可更改为其他命名

示例中的 `PD_DISPATCH_FLOATING_TYPES` 会展开得到 `float32` 与 `float64` 的switch-case实现，从而在运行时根据输入的数据类型，选择实际需要执行的分支。

例如，`ReluCPUForward` 中的 `PD_DISPATCH_FLOATING_TYPES` 实际代码展开如下：

```c++
switch(x.type()) {
  case paddle::DataType::FLOAT32:
    relu_cpu_forward_kernel<float>(
            x.data<float>(), out.mutable_data<float>(x.place()), x.size());
    break;
  case paddle::DataType::FLOAT64:
    relu_cpu_forward_kernel<double>(
            x.data<double>(), out.mutable_data<double>(x.place()), x.size());
    break;
  default:
    PD_THROW(
      "function relu_cpu_forward_kernel is not implemented for data type `",
      paddle::ToString(x.type()), "`");
}
```

目前定义的dispatch宏包括：

- `PD_DISPATCH_FLOATING_TYPES` ：dispatch生成float32和float64对应的实现
- `PD_DISPATCH_INTEGRAL_TYPES` ：dispatch生成int8_t, uint8_t, int16_t, int的int64_t对应的实现
- `PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES` ：dispatch生成前述两个宏全部数据类型对应的实现

当然，如果这几个宏无法满足您实际使用的需求，您可以直接通过switch-case语句实现，将来视需求我们也会添加更多的宏。

#### CPU&CUDA混合实现

通常只有CPU的算子实现是不够的，实际生产环境中一般使用GPU算子。此处将前述 `relu_cpu.cc` 中算子的CPU实现改为GPU示例如下：

- relu_cuda.cu
```c++
#include "paddle/extension.h"

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t* dy,
                                          const data_t* y,
                                          data_t* dx,
                                          const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kGPU);

  out.reshape(x.shape());
  int numel = x.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), numel);
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out) {
  auto grad_x = paddle::Tensor(paddle::PlaceType::kGPU);
  grad_x.reshape(x.shape());

  int numel = out.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x.mutable_data<data_t>(x.place()),
            numel);
      }));

  return {grad_x};
}
```

- relu_cuda.cc
```c++
#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

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

在 `.cu` 文件中实现对应的CUDA kernel和计算函数，在 `.cc` 文件中声明调用即可。

注意这里的 `CHECK_INPUT` 也改为检查输入Tensor是否在GPU上，如果后续仍然在CPU上执行，报错如下：

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

### 维度与类型推导函数

`PaddlePaddle` 框架同时支持动态图与静态图的执行模式，在静态图模式下，组网阶段需要完成Tensor shape和dtype的推导，从而生成正确的模型描述，用于后续Graph优化与执行。因此，除了算子的运算函数之外，还需要实现前向运算的维度和类型的推导函数。

维度推导（InferShape）和类型推导（InferDtype）的函数写法也是有要求的，形式如下：

```c++
std::vector<std::vector<int64_t>> OpInferShape(std::vector<int64_t> x_shape, ...) {
  return {x_shape, ...};
}

std::vector<paddle::DataType> OpInferDtype(paddle::DataType x_dtype, ...) {
  return {x_dtype, ...};
}
```

函数的输入参数与返回值类型固定，如上述代码片段所示，其他要求如下：
- 函数输入参数与前述运算函数的输入Tensor按顺序一一对应，依次为输入参数的shape和dtype
- 函数返回值vector中的shape或dtype信息也需要与返回Tensor按顺序一一对应
- 维度与类型推导函数不支持Attribute的输入


以 `relu` 为例，其维度与类型推导函数如下：

- relu_cpu_fp32.cc / relu_cpu.cc / relu_cuda.cc

```c++
std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> ReluInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}
```

对于仅有一个输入Tensor和一个输出Tensor的自定义算子，如果输出Tensor和输入Tensor的shape和dtype一致，可以省略InferShape和InferDtype函数的实现，其他场景下均需要实现这两个函数。因此，对于这里的 `relu` 算子来说，这两个函数可以不写。

### 构建算子

最后，需要调用 `PD_BUILD_OP` 系列宏，添加算子的描述信息，并关联前述算子运算函数和维度、类型推导函数。

我们提供了3个Build算子的宏：

- `PD_BUILD_OP` ：用于构建前向算子
- `PD_BUILD_GRAD_OP` ：用于构建前向算子对应的反向算子
- `PD_BUILD_DOUBLE_GRAD_OP` ：用于构建前反向算子对应的二次求导算子

对于 `relu` CPU示例来说，构建算子描述如下：

- relu_cpu_fp32.cc / relu_cpu.cc

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
- PD_BUILD_OP系列宏后面的括号内为算子名，也是后面在python端使用的接口名，注意前后不需要引号，注意该算子名不能与 `PaddlePaddle` 内已有算子名重名，比如 `relu` 为 `PaddlePaddle` 内已有算子，如果直接使用relu作为算子名将无法注册成功，所以此处增加了前缀 `custom_`
- Inputs与Outputs的输入参数为 `std::vector<std::string>` ，依次是前面算子运算函数的输入输出Tensor的name，需要按顺序一一对应
- 在声明反向算子的Inputs与Outputs时，前向Tensor对应的梯度Tensor名需要由 `paddle::Grad` 处理前向Tensor名得到，不能够随意声明，例如这里 `"X"` 对应的梯度Tensor名为 `paddle::Grad("X")`
- 此处SetKernelFn、SetInferShapeFn与SetInferDtypeFn中的 `PD_KERNEL` 、`PD_INFER_SHAPE` 、`PD_INFER_DTYPE` 宏用于自动转换并统一函数的签名，不可以省略
- 目前构建反向算子不需要SetInferShapeFn和SetInferDtypeFn

如前述介绍，此处 `relu` 可以省略InferShape和InferDtype函数的实现，因此也可以省略写为：

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

类似地，GPU示例构建算子描述如下：

- relu_cuda.cc

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

#### Attribute声明

对于Attribute的声明，和Inputs、Outputs的声明有所不同，需要按照如下格式声明字符串：

`<name>: <attr-type-expr>`

其中，`name` 为Attribute变量的name，`<attr-type-expr>` 为Attribute变量的类型，类型字符串需要与C++类型严格一致。通过如下示例说明：

假如有前向运算函数形式如下：

```c++
std::vector<paddle::Tensor> AttrTestForward(
    const paddle::Tensor& x,
    bool bool_attr,
    int int_attr,
    float float_attr,
    int64_t int64_attr,
    std::string str_attr,
    std::vector<int> int_vec_attr,
    std::vector<float> float_vec_attr,
    std::vector<int64_t> int64_vec_attr,
    std::vector<std::string> str_vec_attr) {...}
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

如果该算子需要反向实现，反向算子的Attribute输入参数需要是前向算子Attribute输入参数的子集，不能新增前向算子没有的Attribute，示例如下：

```c++
std::vector<paddle::Tensor> AttrTestBackward(
    const paddle::Tensor& grad_out,
    int int_attr,
    std::vector<float> float_vec_attr,
    std::vector<std::string> str_vec_attr) {...}

PD_BUILD_GRAD_OP(attr_test)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"int_attr: int",
            "float_vec_attr: std::vector<float>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestBackward));
```

这里的 `int_attr`、`float_vec_attr`、`str_vec_attr` 均是前向算子声明中出现过的参数。

## 算子编译与使用

本机制提供了两种编译自定义算子的方式，分别为**使用 `setuptools` 编译**与**即时编译**，下面依次通过示例介绍。

### 使用 `setuptools` 编译

该方式是对python 内建库中的 `setuptools.setup` 接口的进一步封装，能够自动地生成 Python API 并以 Module 的形式安装到 site-packages 目录。编译完成后，支持通过 import 语句导入使用。

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

其中 `paddle.utils.cpp_extension.setup` 能够自动搜索和检查本地的 `cc` 和 `nvcc` 编译命令和版本环境，根据用户指定的 `Extension` 类型，完成CPU或CPU设备的算子编译安装。

执行 `python setup_cpu/cuda.py install` 即可一键完成自定义算子的编译和安装。

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
_ops-0.0.0-py3.7-linux-x86_64.egg
custom_setup_ops_pd_.so  EGG-INFO/     relu_cpu.o      relu_cuda.o
custom_setup_ops.py      __pycache__/  relu_cuda.cu.o  version.txt
```

其中 `custom_setup_ops.py` 为根据 `PaddlePaddle` 接口定义规则，自动生成的自定义算子 python 模块源码，其示例内容为（自动生成的代码后续可能会更新）：

```python
import os
import sys
import types
import paddle

def inject_ext_module(module_name, api_names):
    if module_name in sys.modules:
        return sys.modules[module_name]

    new_module = types.ModuleType(module_name)
    for api_name in api_names:
        setattr(new_module, api_name, eval(api_name))

    return new_module

def __bootstrap__():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(cur_dir, "custom_setup_ops_pd_.so")

    assert os.path.exists(so_path)

    # load custom op shared library with abs path
    new_custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
    m = inject_ext_module(__name__, new_custom_ops)

__bootstrap__()

from paddle.fluid.layer_helper import LayerHelper

def custom_relu(x):
    helper = LayerHelper("custom_relu", **locals())

    # prepare inputs and outputs
    ins = {'X' : x}
    attrs = {}
    outs = {}
    out_names = ['Out']
    for out_name in out_names:
        # Set 'float32' temporarily, and the actual dtype of output variable will be inferred
        # in runtime.
        outs[out_name] = helper.create_variable(dtype='float32')

    helper.append_op(type="custom_relu", inputs=ins, outputs=outs, attrs=attrs)

    res = [outs[out_name] for out_name in out_names]

    return res[0] if len(res)==1 else res
```

随后，可以直接在构建模型过程中导入使用，简单示例如下：

```python
import paddle
from custom_setup_ops import custom_relu

x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_relu(x)
```


> 注：`setuptools` 的封装是为了简化自定义算子编译和使用流程，即使不依赖于 `setuptools` ，也可以自行编译生成动态库，并封装相应的python API，然后在基于 `PaddlePaddle` 实现的模型中使用

如果需要详细了解相关接口，或需要配置其他编译选项，请参考以下API文档：

- [paddle.utils.cpp_extension.setup](https://www.paddlepaddle.org.cn/documentation/docs/zh/api//paddle/utils/cpp_extension/setup_cn.html)
- [paddle.utils.cpp_extension.setupCppExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api//paddle/utils/cpp_extension/CppExtension_cn.html)
- [paddle.utils.cpp_extension.CUDAExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api//paddle/utils/cpp_extension/CUDAExtension_cn.html)

### 即时编译（`JIT Compile`）

即时编译将 `setuptools` 编译方式做了进一步的封装，通过将自定义算子对应的 `.cc` 和 `.cu` 文件传入API `paddle.utils.cpp_extension.load`，在后台生成 `setup.py` 文件，并通过子进程的方式，隐式地执行源码文件编译、符号链接、动态库生成、组网 API 接口生成等一系列过程。不需要本地预装 CMake 或者 Ninja 等工具命令，仅需必要的编译器命令环境，如 Linux 下需安装版本不低于 5.4 的 GCC，并软链到 `/usr/bin/cc` ；若编译支持 GPU 设备的算子，则需要预装 `nvcc` 编译环境。

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

`load` 返回一个包含自定义算子API的 `Module` 对象，可以直接使用自定义算子name调用API。

以Linux平台为例，`load` 接口调用过程中，如果不指定 `build_directory` 参数，会默认在 `/root/.cache/paddle_extensions` 目录下先生成一个 `{name}_setup.py`，然后通过subprocess执行 `python {name}_setup.py build`，后续过程与前述 setuptools 编译安装过程一致。

对于本示例，默认生成路径内容如下：

```
λ ls /root/.cache/paddle_extensions/
custom_jit_ops/  custom_jit_ops_setup.py
```

其中，`custom_jit_ops_setup.py` 是生成的setup编译文件，`custom_jit_ops` 目录是编译生成的内容。

如果需要详细了解load接口，或需要配置其他编译选项，请参考API文档 [paddle.utils.cpp_extension.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api//paddle/utils/cpp_extension/load_cn.html) 。

### 同时编译多个算子

以上两种方式均支持同时编译多个自定义算子，只需要将多个算子对应的源文件均传入对应的参数，编译生成的动态库中会包含多个算子的实现，导入 Module 之后，同样以算子名作为API名进行调用，示例如下：

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

- JIT compile
```python
from paddle.utils.cpp_extension import load

custom_ops = load(
    name='custom_jit_ops',
    sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu'])
```

### ABI兼容性检查

以上两种方式，编译前均会执行 ABI 兼容性检查 ，即检查 cc 命令对应的 GCC 版本是否与编译本地安装的 `PaddlePaddle` 时的 GCC 版本一致。如对于 CUDA 10.1 以上的 `PaddlePaddle` 默认使用 GCC 8.2 编译，则本地 cc 对应的编译器版本也需为 8.2 ，否则可能由于 ABI 兼容性原因引发自定义 OP 执行时报错。

> 注：编译器的 ABI 兼容性是向前兼容的，Linux 下推荐使用 GCC 8.2 高版本作为 `/usr/bin/cc` 命令的软链对象

## 算子验证

经过前述过程，自定义算子的编写、编译安装及API生成均已完成，接下来需要对自定义算子在动态图和静态图模式下进行可用性验证。

以下验证用例均基于前述源文件 `relu_cuda.cc` 和 `relu_cuda.cu` 测试 `custom_relu` 在GPU环境中的使用，均采用JIT Compile的方式编译自定义算子。

通过定义一个简单的网络模型，完成训练和存储推理模型的基本过程。

### 动态图模式验证

动态图模式的使用示例如下：

```python
import numpy as np
import paddle
import paddle.nn as nn
from paddle.utils.cpp_extension import load

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class Net(nn.Layer):
    """
    A simple example for Regression Model.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, 100)
        self.fc2 = nn.Linear(100, CLASS_NUM)

    def forward(self, x):
        tmp1 = self.fc1(x)
        # call custom relu op
        tmp_out = custom_ops.custom_relu(tmp1)
        tmp2 = self.fc2(tmp_out)
        # call custom relu op
        out = custom_ops.custom_relu(tmp2)
        return out

paddle.set_device("gpu")

# create network
net = Net()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(loader()):
        out = net(image)
        loss = loss_fn(out, label)
        loss.backward()

        opt.step()
        opt.clear_grad()
        print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

# save inference model
path = "custom_relu_dynamic/net"
paddle.jit.save(net, path,
    input_spec=[paddle.static.InputSpec(shape=[None, 784], dtype='float32')])
```

### 静态图模式验证

静态图模式的使用示例如下：

```python
import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static
from paddle.utils.cpp_extension import load

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

# switch static mode
paddle.enable_static()
paddle.set_device("gpu")

# create network
image  = static.data(shape=[None, IMAGE_SIZE], name='image', dtype='float32')
label = static.data(shape=[None, 1], name='label', dtype='int64')

tmp1 = static.nn.fc(x=image, size=100)
tmp_out = custom_ops.custom_relu(tmp1)
tmp2 = static.nn.fc(x=tmp_out, size=CLASS_NUM)
out = custom_ops.custom_relu(tmp2)

loss = nn.functional.cross_entropy(out, label)
opt = paddle.optimizer.SGD(learning_rate=0.01)
opt.minimize(loss)

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# prepare
exe = static.Executor()
exe.run(static.default_startup_program())

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image_data, label_data) in enumerate(loader()):
        loss_data = exe.run(static.default_main_program(),
            feed={'image': image_data,
                  'label': label_data},
            fetch_list=[loss])
        print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss_data)))

# save inference model
path = "custom_relu_static/net"
static.save_inference_model(path, [image], [out], exe)
```

## 算子在推理场景中的使用

基于本机制编写的自定义算子，能够在推理场景中使用，仍然基于前述示例介绍使用流程，这里基于 `relu_cuda.cc` 和 `relu_cuda.cu` 介绍。

### 源码改动

由于接口管理上存在一些差别，自定义算子源码中的引入的头文件需要替换一下：

`#include "paddle/extension.h"`

改为

`#include "paddle/include/experimental/ext_all.h"`

其他地方不需要做改动。

### 自定义算子与推理库联合编译

编写推理的测试程序，其中需要使用前述验证过程中存储的inference model，目录为 `custom_relu_dynamic/net` 或者 `custom_relu_static/net` ，具体示例如下：

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

编写 `CMakeList` 编译构建文件，示例如下：

```cmake
cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    ON)
option(USE_TENSORRT "Compile demo with TensorRT."   ON)
option(CUSTOM_OPERATOR_FILES "List of file names for custom operators" "")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(external/boost)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${FLAG_OPENMP}")
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

编写执行执行脚本 `run.sh` ，示例如下：

```cmake
mkdir -p build
cd build
rm -rf *

DEMO_NAME=custom_op_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=/shixiaowei02/Paddle-custom-op-src/Paddle/build/paddle_inference_install_dir
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

注意要根据实际情况对执行脚本中的几处配置进行调整：

```cmake
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON  
WITH_GPU=ON  
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/paddle/nvidia-downloads/cudnn_v7.5_cuda10.1/lib64
CUDA_LIB=/paddle/nvidia-downloads/cuda-10.1/lib64
# TENSORRT_ROOT=/paddle/nvidia-downloads/TensorRT-6.0.1.5
```

然后，运行 `sh run.sh` ，完成编译，会在目录下产生build目录。

### 运行推理程序

```
# 进入build目录
cd build
# 运行样例
./custom_op_test
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

### 更多推理参考文档

- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
