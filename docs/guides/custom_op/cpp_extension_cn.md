# 自定义 C++ 扩展

## 概述

飞桨框架提供了丰富的算子库，能够满足绝大多数场景的使用需求。但是在以下场景下，您可能希望定制化 C++ 实现，从而满足特定需求：

1. python 函数经常被调用，需要对性能做优化；
2. 即使该函数被调用的次数很少，但是该函数开销很重，希望将实现逻辑放到 C++ 端提高性能；
3. 函数实现依赖或者需要调用 C++ 库。

为此，我们提供了 C++ 扩展机制，以此机制实现的自定义扩展，能够 **即插即用** ，不需要重新编译安装飞桨框架。

使用 C++ 扩展机制，仅需要以下两个步骤：

1. 实现 C++ 扩展的运算函数，将该函数与 python 端绑定
2. 调用 `python` 接口完成 C++ 扩展的编译与注册

随后即可在模型中使用，下面通过实现一个自定义的加法运算，介绍具体的实现、编译与应用流程。

> 注意事项：
>
> - 在使用本机制实现 C++ 扩展之前，请确保已经正确安装了 `PaddlePaddle develop` 版本
> - 本机制目前仅支持 `Linux` 平台。
> - 本机制不支持动转静以及推理部署。

## C++ 扩展实现

### 基本写法要求

在编写运算函数之前，需要引入 `PaddlePaddle` 扩展头文件，示例如下：

```c++
#include "paddle/extension.h"
```

`extension.h` 头文件包含两方面内容：

1. phi 算子库，包括 C++ 端对 Tensor 进行计算的 API
2. pybind11 库，绑定 python 和 C++ 端实现

下面结合具体的示例，介绍 C++ 扩展的实现方式。

### C++扩展实现

以一个自定义的加法操作 `custom_add` 和自定义求幂结构体 `Power` 为例，C++ 扩展实现如下：

- custom_power.h

```c++
#pragma once

#include "paddle/extension.h"

struct Power {
  Power(int A, int B) {
    tensor_ = paddle::ones({A, B}, phi::DataType::FLOAT32, phi::CPUPlace());
  }
  explicit Power(paddle::Tensor x) { tensor_ = x; }
  paddle::Tensor forward() { return tensor_.pow(2); }
  paddle::Tensor get() const { return tensor_; }

 private:
  paddle::Tensor tensor_;
};
```

- custom_add.cc

```c++
#include "paddle/extension.h"

#include "custom_power.h"  // NOLINT

// 自定义的加法实现，out = exp(x) + exp(y)
paddle::Tensor custom_add(const paddle::Tensor& x, const paddle::Tensor& y) {
  return paddle::add(paddle::exp(x), paddle::exp(y));
}

PYBIND11_MODULE(custom_cpp_extension, m) {
  // 将加法函数绑定至 python 端
  m.def("custom_add", &custom_add, "exp(x) + exp(y)");

  // 将 Power 类绑定至 python 端
  py::class_<Power>(m, "Power")
      .def(py::init<int, int>())
      .def(py::init<paddle::Tensor>())
      .def("forward", &Power::forward)
      .def("get", &Power::get);
}
```

主要逻辑包括：

1. 定义 C++ 扩展的实现逻辑，例如本示例中的 `custom_add` 函数
2. 使用 pybind11 将加法函数绑定至 python 端，pybind11 使用详情可以参考 [pybind11 文档](https://pybind11.readthedocs.io/en/stable/)

## C++扩展的编译与使用

本机制提供了两种编译自定义算子的方式，分别为 **使用 `setuptools` 编译** 与 **即时编译** ，下面依次通过示例介绍。

### 使用 `setuptools` 编译

该方式是对 `python` 内建库中的 `setuptools.setup` 接口的进一步封装，能够自动地以 Module 的形式安装到 site-packages 目录。编译完成后，支持通过 import 语句导入使用。

您需要编写 `setup.py` 文件， 配置 C++ 扩展的编译规则。

例如，前述 `custom_add.cc` 示例的 `setup` 文件可以实现如下：

- setup_custom_add.py ( for custom_add.cc )

```python
from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='custom_cpp_extension',
    ext_modules=CppExtension(
        sources=['custom_add.cc']
    )
)
```

其中 `paddle.utils.cpp_extension.setup` 能够自动搜索和检查本地的 `cc(Linux)` 编译命令和版本环境，完成 CPU 设备的 C++ 扩展编译安装。

执行 `python setup_custom_add.py install` 即可一键完成 C++ 扩展的编译和安装。

> 注意：`setup` 参数中 `name` 的值应与 `PYBIND11_MODULE` 宏声明的模块名一致

安装完成后，可以通过 help() 函数来查看相应的函数签名

```python-repl
help(custom_cpp_extension.custom_add)
```

对应的输出为：

```python
Help on built-in function custom_add in module custom_cpp_extension:

custom_add(...) method of builtins.PyCapsule instance
    custom_add(arg0: paddle::experimental::Tensor, arg1: paddle::experimental::Tensor) -> paddle::experimental::Tensor

    exp(x) + exp(y)

```

随后，可以直接在构建模型过程中导入使用，简单示例如下：

```python
import paddle
from custom_cpp_extension import custom_add

x = paddle.randn([4, 10], dtype='float32')
y = paddle.randn([4, 10], dtype='float32')
out = custom_add(x, y)
```

> 注：`setuptools` 的封装是为了简化 C++ 扩展的编译和使用流程，即使不依赖于 `setuptools` ，也可以自行编译生成动态库，并封装相应的 python API，然后在基于 `PaddlePaddle` 实现的模型中使用

如果需要详细了解相关接口，或需要配置其他编译选项，请参考以下 API 文档：

- [paddle.utils.cpp_extension.setup](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/setup_cn.html)
- [paddle.utils.cpp_extension.setupCppExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CppExtension_cn.html)

### 即时编译（`JIT Compile`）

即时编译将 `setuptools.setup` 编译方式做了进一步的封装，通过将 C++ 扩展对应的 `.cc` 文件传入 API `paddle.utils.cpp_extension.load`，在后台生成 `setup.py` 文件，并通过子进程的方式，隐式地执行源码文件编译、符号链接、动态库生成等一系列过程。不需要本地预装 CMake 或者 Ninja 等工具命令，仅需必要的编译器命令环境。 Linux 下需安装版本不低于 5.4 的 GCC，并软链到 `/usr/bin/cc`

对于前述 `custom_add.cc` 示例，使用方式如下：

- for custom_add.cc

```python
import paddle
from paddle.utils.cpp_extension import load

custom_cpp_extension = load(
    name="custom_cpp_extension",
    sources=['custom_add.cc'])

x = paddle.randn([4, 10], dtype='float32')
y = paddle.randn([4, 10], dtype='float32')
out = custom_cpp_extension.custom_add(x, y)
```

`load` 返回一个 `Module` 对象，可以直接使用 C++ 扩展名调用 API。

> 注意：`load` 参数中 `name` 的值应与 `PYBIND11_MODULE` 宏声明的模块名一致

`load` 接口调用过程中，如果不指定 `build_directory` 参数，Linux 会默认在 `~/.cache/paddle_extensions` 目录下生成一个 `{name}_setup.py` 文件，然后通过 subprocess 执行 `python {name}_setup.py build`，然后载入动态库，生成 Python API 之后返回。

对于本示例，默认生成路径内容如下：

```
ls ~/.cache/paddle_extensions/
custom_jit_ops/  custom_jit_ops_setup.py
```

其中，`custom_jit_ops_setup.py` 是生成的 setup 编译文件，`custom_jit_ops` 目录是编译生成的内容。

如果需要详细了解 load 接口，或需要配置其他编译选项，请参考 API 文档 [paddle.utils.cpp_extension.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/load_cn.html) 。

### ABI 兼容性检查

以上两种方式，编译前均会执行 ABI 兼容性检查 。对于 Linux，会检查 cc 命令对应的 GCC 版本是否与所安装的 `PaddlePaddle` 的 GCC 版本一致。例如对于 CUDA 10.1 以上的 `PaddlePaddle` 默认使用 GCC 8.2 编译，则本地 cc 对应的编译器版本也需为 8.2。如果上述版本不一致，则会打印出相应 warning，且可能引发 C++ 扩展编译执行报错。
