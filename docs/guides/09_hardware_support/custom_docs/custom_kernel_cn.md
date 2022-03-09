# 自定义外部Kernel

## 概述

Kernel函数（简称Kernel）对应算子（Operator，简称Op）的具体实现，飞桨框架提供了丰富的Op和不同硬件（如CPU、GPU、XPU等）上的Kernel实现。对于通过自定义Runtime机制注册的外部硬件，飞桨提供了配套的自定义外部Kernel机制，实现独立于框架的Kernel编码与注册。

与[自定义原生算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_op_cn.html)、[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op_cn.html)不同，自定义外部Kernel有如下特点：

1. 框架解耦的Kernel实现、编译与安装
2. 内外一致的Kernel声明、编码与注册
3. 方便的互调用性，避免重复逻辑实现

其使用方式为：

1. 确定Kernel的函数声明
2. 实现Kernel函数体
2. 注册自定义外部Kernel
4. 独立编译、安装与飞桨自动加载

随后即可在模型中使用，下面通过实现[昇腾NPU硬件](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/npu_docs/index_cn.html)的自定义kernel `softmax` ，介绍其具体的实现、编译与应用流程。其中`softmax`的功能可参考[Paddle官网API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softmax_cn.html#softmax)。

> 注意事项：
> - 在使用本机制实现自定义Kernel之前，请确保已经正确安装了[飞桨develop](https://github.com/PaddlePaddle/Paddle)最新版本
> - 当前仅支持 `Linux`平台
> - 仅支持Phi算子库已开放Kernel声明的自定义编码与注册

## Kernel函数声明

Kernel函数声明是飞桨通过头文件发布的Kernel函数约定，框架内外一致。

在编写具体的Kernel函数前，按需查找飞桨开放的头文件以确定待实现具体Kernel的函数声明。头文件位于飞桨安装路径的`include/paddle/phi/kernels/`下。

如`softmax`的Kernel函数位于`softmax_kernel.h`中，具体如下：
```c++
// paddle/phi/kernels/softmax_kernel.h
// ...

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DataType dtype,
                   DenseTensor* out) {
  // ...  
}
// ...
```
相关约定为：

1. 模板参数：固定写法，第一个模板参数为数据类型`T`，第二个模板参数为设备上下文`Context`。
2. 函数返回：固定为`void`。
3. 函数命名：Kernel名称+Kernel后缀，驼峰式命名，如`SoftmaxKernel`。
4. 函数参数：依次为`Context`，`InputTensor`，`Attribute`和`OutTensor*`。其中：
- 首位是设备上下文参数，固定为`const Context&`类型；
    - 自定义外部Kernel对应`CustomContext`类型，具体可参照[custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h)
- 其次是输入的`Tensor`参数，数量>=0，支持的类型包括：
    - `const DenseTensor&` 具体参照[dense_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/dense_tensor.h)
    - `const SelectedRows&`具体参照[selected_rows.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/selected_rows.h)
    - `const SparseCooTensor&`具体参照[sparse_coo_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_coo_tensor.h)
    - `const SparseCsrTensor&`具体参照[sparse_csr_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_csr_tensor.h)
    - `const std::vector<DenseTensor*>&`
    - `const std::vector<SparseCooTensor*>&`
    - `const std::vector<SparseCsrTensor*>&`
- 然后是输入的`Attribute`参数，数量>=0，支持的类型包括：
    - `bool`
    - `float`
    - `double`
    - `int`
    - `int64_t`
    - `phi::dtype::float16` 具体参照[float16.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float16.h)
    - `const Scalar&` 具体参照[scalar.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/scalar.h)
    - `DataType` 具体参照[data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
    - `DataLayout` 具体参照[layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
    - `Place` 具体参照[place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
    - `const std::vector<int64_t>&`
    - `const ScalarArray&` 具体参照[scalar_array.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/scalar_array.h)
    - `const std::vector<int>&`
    - `const std::string&`
    - `const std::vector<bool>&`
    - `const std::vector<float>&`
    - `const std::vector<double>&`
    - `const std::vector<std::string>&`
- 最后是输出的`Tensor*`参数，数量>0，支持的类型包括：
    - `DenseTensor*`
    - `SelectedRows*`
    - `SparseCooTensor*`
    - `SparseCsrTensor*`
    - `std::vector<DenseTensor*>`
    - `std::vector<SparseCooTensor*>`
    - `std::vector<SparseCsrTensor*>`

需要注意的是，在飞桨开放的Kernel头文件中，如果Kernel函数已经通过调用其它函数或者更底层的Kernel函数实现，可逐层向下实现其调用的函数。通过这种方式，可以尽可能复用Kernel函数以减少重复代码。

如`softmax_kernel.h`中，`SoftmaxKernel`通过调用`Cast`与`SoftmaxRawKernel`实现功能：

```c++
// ...
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out);

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DataType dtype,
                   DenseTensor* out) {
  auto cast_x = phi::Cast<T, Context>(dev_ctx, x, dtype);
  phi::SoftmaxRawKernel<T, Context>(dev_ctx, axis, out);
}

}  // namespace phi
```

而`Cast`函数通过调用`CastKernel`实现：

```c++
// ...
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out);

template <typename T, typename Context>
DenseTensor Cast(const Context& dev_ctx,
                 const DenseTensor& x,
                 DataType out_dtype) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  CastInferMeta(x, out_dtype, &meta_out);
  CastKernel<T, Context>(dev_ctx, x, out_dtype, &dense_out);
  return dense_out;
}

}  // namespace phi
```
所以，为实现`SoftmaxKernel`，需实现`SoftmaxRawKernel`和`CastKernel`。通过这种方式，类似`CastKernel`的基础Kernel仅需实现一次便可被其它Kernel调用。二者的Kernel函数声明如下：

```c++
// CastKernel
template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out);
```
```c++
// SoftmaxRawKernel
template <typename T, typename Context>
void SoftmaxRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out);
```
> 注意：
> 1. Kernel函数声明是自定义外部Kernel能够被注册和框架调用的基础，由框架发布，需要严格遵守
> 2. Kernel函数与头文件可能不完全对应，可以按照函数命名约定等查找所需Kernel函数声明
> 3. 推荐逐层向下查找和实现自定义Kernel，这样能够充分复用与避免重复，从而降低开发成本

## Kernel函数实现

Kernel函数实现根据Kernel函数声明和Kernel功能，基于飞桨API和硬件封装库API完成具体的逻辑实现。

### 基本写法

在实际编写Kernel函数体之前，需要引入飞桨发布的头文件：

```c++
#include "paddle/phi/extension.h"
```

该文件统一包含了飞桨对外发布用于自定义外部Kernel开发所必须的头文件。包括前文描述的Kernel函数声明、Kernel参数中各种数据类型等。

> 注：也可以只引入更细层次的必要头文件

在自定义外部Kernel场景中，用户一般基于硬件自身封装库的API进行逻辑实现，所以也需要引入相应库的头文件，如NPU自定义Kernel通过调用`libascendcl.so`完成逻辑计算：
```c++
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
```

然后，便可以基于飞桨开放的API和硬件封装库的API进行Kernel函数体的编码。

其中，飞桨开放的API分布在开放的头文件中，简要介绍如下：

#### Context API
- 设备信息获取API：
    - `void* stream() const` 返回void*类型的`stream`
    - `const Place& GetPlace()` 返回当前设备的`Place`
- 设备内存分配API：
    - `template <typename T> T* Alloc(TensorBase* tensor, size_t requested_size = 0) const` 为给定的tensor指针分配内存

#### Tensor API
- Tensor信息获取：
    - `int64_t numel() const` 返回Tensor元素数量
    - `const DDim& dims() const` 返回Tensor的dims信息
    - `DataType dtype() const` 返回Tensor元素的数据类型
    - `DataLayout layout() const` 返回Tensor的layout信息
    - `const Place& place() const` 返回Tensor的place
- Tensor操作：
    - `void set_meta(DenseTensorMeta&& meta)` 设置Tensor的Meta信息
    - `DenseTensor& Resize(const DDim& dims)` 修改Tensor的dims
    - `DenseTensor& ShareDataWith(const DenseTensor& src)`两个Tensor共享相同内存

#### Exception API
飞桨开放了封装的Exception API方便异常判断，具体参照[enforce.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/enforce.h)与[errors](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/errors.h)：

基本使用方式为：
```
 PADDLE_ENFORCE_TYPE(cond_a, // 条件A
                     cond_b, // 条件B, 根据TYPE可选
                     phi::errors::ERR_TYPE("ERR_MSG"));
```

其中根据`TYPE`的不同，支持的判断包括：

- `PADDLE_ENFORCE_EQ`：cond_a == cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_NE`：cond_a != cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_GT`：cond_a > cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_GE`：cond_a >= cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_LT`：cond_a < cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_LE`：cond_a <= cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_NOT_NULL`：cond_a != nullptr，否则触发ERR_TYPE异常和报ERR_MSG

其中配合使用的ERR_TYPE支持包括：

- `InvalidArgument`：非法参数
- `NotFound`：未找到
- `OutOfRange`：越界
- `AlreadyExists`：已存在
- `ResourceExhausted`：资源超限
- `PreconditionNotMet`：前置条件未满足
- `PermissionDenied`：权限限制
- `ExecutionTimeout`：超时
- `Unimplemented`：未实现
- `Unavailable`：不可用
- `Fatal`：Fatal错误
- `External`：外部错误

其中ERR_MSG为C语言风格字符串，支持变长参数。

Exception API使用举例如下：

```c++
PADDLE_ENFORCE_EQ(
      (num_col_dims >= 2 && num_col_dims <= src.size()),
      true,
      phi::errors::InvalidArgument("The num_col_dims should be inside [2, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size(),
                                   num_col_dims));
```
当num_col_dims >= 2 && num_col_dims <= src.size()不为true时，报非法参数错误并输出报错信息。

>注：Kernel函数实现可用的飞桨开放API众多，无法一一列出，但框架内外使用一致，更详细的API用法请按需参照相应头文件和[飞桨框架](https://github.com/PaddlePaddle/Paddle)内的使用

### 函数实现

基于以上飞桨开放API和`libascendcl.so`的API，`SoftmaxRawKernel`的实现如下：

```c++
#include "npu_op_runner.h" // 基于ascendcl库封装NpuOpRunner

namespace custom_kernel {
template <typename T, typename Context>
void SoftmaxRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out) {
    std::vector<int> axes;
    axes.push_back(axis);
    dev_ctx.template Alloc<T>(out);
    const auto& runner = NpuOpRunner("SoftmaxV2", {x}, {*out}, {{"axes", axes}});
    aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
    runner.Run(stream);
}
} // namespace custom_kernel
```

由于昇腾NPU通过`ascendcl`库对外封装了其支持的算子库，所以此处NPU的`SoftmaxKernel`实现的基本原理为根据飞桨Kernel函数的输入、属性和输出映射调用NPU算子的输入、输出和属性，然后通过封装的NPU算子调用代理`NpuOpRunner`调用NPU的算子完成逻辑计算。

在函数体实现中，基于设备上下文参数为`dev_ctx`调用`Alloc`API为输出Tensor参数`out`分配片上内存，并获取NPU硬件`stream`便于调用NPU算子。

类似`SoftmaxKernel`，softmax的反向Kernel函数通过确定其对应的Kernel函数声明，实现如下：
```
template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
    auto dims = x_grad->dims();
    const int rank = dims.size();
    const int use_axis = axis < 0 ? axis + rank : axis;
    int64_t first_dim = 1;
    int64_t sec_dim = 1;
    for (int i = 0; i < use_axis; i++) {
      first_dim *= dims[i];
    }
    for (int i = use_axis; i < rank; i++) {
      sec_dim *= dims[i];
    }

    phi::DenseTensor tmp_out;
    tmp_out.ShareDataWith(out).Resize({first_dim, sec_dim});

    phi::DenseTensor tmp_out_grad;
    tmp_out_grad.ShareDataWith(out_grad).Resize({first_dim, sec_dim});

    x_grad->Resize(phi::make_ddim({first_dim, sec_dim}));
    dev_ctx.template Alloc<T>(x_grad);

    const auto& runner = NpuOpRunner("SoftmaxGrad",
                                     {tmp_out, tmp_out_grad}, {*x_grad}, {});

    aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
    runner.Run(stream);

    x_grad->Resize(dims);
}

```
在`SoftmaxGradKernel`实现函数体中，基本原理仍是Kernel函数参数和NPU算子输入、输出和属性的映射，区别是更多地使用了Tensor操作API，如`dims()`，`Resize()`与`ShareDataWith()`。

>注：针对不同硬件实现其自定义Kernel的方式可能不同，需要结合实际硬件进行设计，但飞桨开放API的使用相同。

## Kernel函数注册

在完成Kernel函数体的实现后，需要通过自定义外部Kernel的注册宏进行Kernel注册以便飞桨框架调用。

注册宏的基本形式：具体参照[kernel_registry.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/kernel_registry.h)

```
PD_REGISTER_PLUGIN_KERNEL(kernel_name, backend, layout, meta_kernel_fn, ...)) {}
```

说明如下：

- 注册宏名称：固定为`PD_REGISTER_PLUGIN_KERNEL`
- 第一个参数：固定为Kernel名称，飞桨内外一致，可参照CPU相同Kernel函数注册名称，如`softmax`
- 第二个参数：自定义的`backend`名称，须与自定义Runtime设定的名称一致，如`Ascend910`
- 第三个参数：`DataLayout`类型的枚举，须按需选择设定，具体参照[layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
- 第四个参数：固定为Kernel函数名，如`my_namespace::SoftmaxKernel`，注意此处不加模板参数
- 不定长数据类型参数：为C++的基础数据类型或飞桨定义的`float16`、`bfloat16`、`complex`类型
- 末尾：固定为函数体，其中可按需进行必要的设置，如果没有，保留`{}`。

对末尾函数体的补充说明：其对应的函数声明如下：
```
void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(
      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
```
即函数体中可使用`const KernelKey&`类型的`kernel_key`与`Kernel`类型的参数`kernel`，用于特定Kernel注册时对Kernel的个性化调整。如Input与Output数据类型或Layout的调整。

注册宏的位置需要放置在全局空间下。

针对已实现的`SoftmaxRawKernel`的注册如下：

```c++
// 全局空间下
PD_REGISTER_PLUGIN_KERNEL(softmax,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxRawKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
```

注册的kernel_name为`softmax`
自定义的backend名称为`Ascend910`
layout为`ALL_LAYOUT`
meta_kernel_fn为`custom_kernel::SoftmaxRawKernel`
三个数据类型，分别是`float`，`double`和`phi::dtype::float16`
末尾无需针对kernel参数进行调整，所以留空函数体


## Kernel编译与使用

针对外部硬件，自定义外部Kernel的编译和使用与自定义Runtime绑定。

在具体编译时，需要：

1. 保持注册使用的backend参数与自定义Runtime中注册的名称一致
2. 编译需使用第三方库如`boost`，`gflag`和`glog`等

NPU实现demo的`CMakeLists.txt`如下，可重点关注其中`WITH_KERNELS`部分：

```python
cmake_minimum_required(VERSION 3.10)

project(paddle-ascend910 CXX C)

option(WITH_TESTING    "compile plugin with unit testing"        OFF)
option(WITH_KERNELS    "build custom kernels"                    OFF)

set(PLUGIN_NAME        "paddle_ascend910")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_PLUGIN_DIR  "/opt/conda/lib/python3.7/site-packages/paddle-plugins/")

set(PADDLE_INC_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")

set(NPU_INC_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/")
set(NPU_LIB_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/")

set(PLUGIN_SRCS runtime/runtime.cc)
set(INCLUDE_DIR ${PADDLE_INC_DIR} ${NPU_INC_DIR})

if (WITH_KERNELS)
  set(BOOST_INC_DIR      "/workspace/dev/Paddle/build/third_party/boost/src/extern_boost")
  set(GFLAGS_INC_DIR     "/workspace/dev/Paddle/build/third_party/install/gflags/include")
  set(GLOG_INC_DIR       "/workspace/dev/Paddle/build/third_party/install/glog/include")
  set(THREAD_INC_DIR     "/workspace/dev/Paddle/build/third_party/threadpool/src/extern_threadpool")

  set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${THREAD_INC_DIR})
  set(RUNTIME_INC_DIR ${CMAKE_SOURCE_DIR}/runtime)
  list(APPEND INCLUDE_DIR ${THIRD_PARTY_INC_DIR} ${RUNTIME_INC_DIR})

  file(GLOB PLUGIN_KERNEL_FILES RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
  list(APPEND PLUGIN_SRCS ${PLUGIN_KERNEL_FILES})

  find_file(CORE_AVX_FOUND core_avx.so ${PADDLE_LIB_DIR})
  if (CORE_AVX_FOUND)
    set(CORE_LIB ":core_avx.so")
  else()
    set(CORE_LIB ":core_noavx.so")
  endif()

  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)   # for NPU
  add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext
  add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid
endif()

include_directories(${INCLUDE_DIR})
link_directories(${PADDLE_LIB_DIR} ${NPU_LIB_DIR})

####### build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
target_link_libraries(${PLUGIN_NAME} PRIVATE ascendcl)
if (WITH_KERNELS)
 target_link_libraries(${PLUGIN_NAME} PRIVATE ${CORE_LIB})  # special name
endif()

install(TARGETS ${PLUGIN_NAME}
PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
DESTINATION "${PADDLE_PLUGIN_DIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

####### packing wheel package
add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

```
其中：

- THIRD_PARTY_INC_DIR中引入Kernel实现所基于的飞桨开放头文件依赖的三方库，此处使用与飞桨一致
- CORE_LIB用于查找和链接飞桨框架lib

> 注意：
> 1. 添加预定义宏`PADDLE_WITH_CUSTOM_DEVICE`以支持CustomContext
> 2. 添加预定义宏`PADDLE_WITH_CUSTOM_KERNEL`以隔离部分内部未开放的代码，未来会移除
> 3. 对于第三方库的依赖，建议使用与飞桨框架依赖的相同版本

编译后，得到产出为：

```
dist/paddle_ascend910-0.0.1-cp37-cp37m-linux_x86_64.whl
```

安装后，在Python安装路径paddle-plugins下新增`libpaddle_ascend910.so`动态库。

```
(base) λ yq01-sys-rpm0132b55 /workspace/dev/ ls /opt/conda/lib/python3.7/site-packages/paddle-plugins/
libpaddle_ascend910.so
```

此时加载Paddle可自动完成自定义Kernel的加载：
```python
(base) λ yq01-sys-rpm0132b55 /workspace/dev python
Python 3.7.9 (default, Aug 31 2020, 12:42:55)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import paddle
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
I0309 20:36:00.418833 58202 init.cc:259] ENV [CUSTOM_DEVICE_ROOT]=/opt/conda/lib/python3.7/site-packages/paddle-plugins
I0309 20:36:00.418880 58202 init.cc:147] Try loading custom device libs from: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0309 20:36:00.426144 58202 custom_device.cc:711] Successed in loading custom runtime in lib: /opt/conda/lib/python3.7/site-packages/paddle-plugins/libpaddle_ascend910.so
I0309 20:36:00.426468 58202 custom_kernel.cc:65] Successed in loading custom kernels.
I0309 20:36:00.426493 58202 init.cc:159] Finished in LoadCustomDevice with libs_path: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0309 20:36:00.426545 58202 init.cc:265] CustomDevice: Ascend910, visible devices count: 1
>>> paddle.fluid.core._get_all_register_op_kernels('phi')['softmax']
['data_type[float]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]', 'data_type[float]:data_layout[Undefined(AnyLayout)]:place[Place(cpu)]:library_type[PLAIN]', 'data_type[::paddle::platform::float16]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]', 'data_type[double]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]', 'data_type[double]:data_layout[Undefined(AnyLayout)]:place[Place(cpu)]:library_type[PLAIN]']
>>>
```
通过飞桨接口可见`softmax`中已注册`Ascend910`的Kernel，具体如下：

`data_type[float]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]`
`data_type[::paddle::platform::float16]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]`
`data_type[double]:data_layout[Undefined(AnyLayout)]:place[Place(Ascend910:0)]:library_type[PLAIN]`

与预期一致。

通过选定自定义Runtime调用实现的[飞桨官网API文档softmax的代码示例](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softmax_cn.html#softmax)如下：
```
(base) λ yq01-sys-rpm0132b55 /workspace/dev python
Python 3.7.9 (default, Aug 31 2020, 12:42:55)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import paddle
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
I0309 20:45:27.061661 58370 init.cc:259] ENV [CUSTOM_DEVICE_ROOT]=/opt/conda/lib/python3.7/site-packages/paddle-plugins
I0309 20:45:27.061728 58370 init.cc:147] Try loading custom device libs from: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0309 20:45:27.072557 58370 custom_device.cc:711] Successed in loading custom runtime in lib: /opt/conda/lib/python3.7/site-packages/paddle-plugins/libpaddle_ascend910.so
I0309 20:45:27.073024 58370 custom_kernel.cc:65] Successed in loading custom kernels.
I0309 20:45:27.073104 58370 init.cc:159] Finished in LoadCustomDevice with libs_path: [/opt/conda/lib/python3.7/site-packages/paddle-plugins]
I0309 20:45:27.073179 58370 init.cc:265] CustomDevice: Ascend910, visible devices count: 1
>>> import numpy as np
>>> paddle.set_device('Ascend910')
Place(Ascend910:0)
>>> x = np.array([[[2.0, 3.0, 4.0, 5.0],
...                 [3.0, 4.0, 5.0, 6.0],
...                 [7.0, 8.0, 8.0, 9.0]],
...                 [[1.0, 2.0, 3.0, 4.0],
...                 [5.0, 6.0, 7.0, 8.0],
...                 [6.0, 7.0, 8.0, 9.0]]], 'float32')
>>> x = paddle.to_tensor(x)
>>> x
Tensor(shape=[2, 3, 4], dtype=float32, place=Place(Ascend910:0), stop_gradient=True,
       [[[2., 3., 4., 5.],
         [3., 4., 5., 6.],
         [7., 8., 8., 9.]],

        [[1., 2., 3., 4.],
         [5., 6., 7., 8.],
         [6., 7., 8., 9.]]])
>>> out1 = paddle.nn.functional.softmax(x)
>>> out1
Tensor(shape=[2, 3, 4], dtype=float32, place=Place(Ascend910:0), stop_gradient=True,
       [[[0.03205860, 0.08714432, 0.23688281, 0.64391422],
         [0.03205860, 0.08714432, 0.23688281, 0.64391422],
         [0.07232948, 0.19661193, 0.19661193, 0.53444666]],

        [[0.03205860, 0.08714432, 0.23688281, 0.64391422],
         [0.03205860, 0.08714432, 0.23688281, 0.64391422],
         [0.03205860, 0.08714432, 0.23688281, 0.64391422]]])
>>> out2 = paddle.nn.functional.softmax(x, dtype='float64')
>>> out2
Tensor(shape=[2, 3, 4], dtype=float64, place=Place(Ascend910:0), stop_gradient=True,
       [[[0.03205860, 0.08714432, 0.23688282, 0.64391426],
         [0.03205860, 0.08714432, 0.23688282, 0.64391426],
         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],

        [[0.03205860, 0.08714432, 0.23688282, 0.64391426],
         [0.03205860, 0.08714432, 0.23688282, 0.64391426],
         [0.03205860, 0.08714432, 0.23688282, 0.64391426]]])
>>>

```

以上，通过[昇腾NPU硬件](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/npu_docs/index_cn.html)的自定义kernel `softmax` ，介绍了自定义外部Kernel的实现、编译与应用流程。
