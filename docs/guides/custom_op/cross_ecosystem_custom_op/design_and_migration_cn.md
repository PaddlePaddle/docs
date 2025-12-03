# 原理和迁移方式

## 实现原理

为了方便 PyTorch 自定义算子快速接入 PaddlePaddle 框架，我们提供了如下图所示的兼容机制：

<figure align="center">
    <img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/custom_op/cross_ecosystem_custom_op/images/cross-ecosystem-custom-op-compatible.drawio.png?raw=true" width="700" alt='missing' align="center"/>
    <figcaption><center>跨生态自定义算子兼容机制示意图</center></figcaption>
</figure>

正如图上所示，我们自底向上提供了如下几层支持：

- **C++ API 兼容层**：该层实现了常用 PyTorch C++ API 的兼容接口，用户仍然可以通过调用 PyTorch 风格的 `at::*`、`torch::*`、`c10::*` 等命名空间下的函数和类来实现自定义算子逻辑，从而最大限度地复用现有代码，使迁移工作量降至最低。
- **算子注册兼容层**：对于使用 pybind11 进行算子注册的 PyTorch 自定义算子，PaddlePaddle 无需额外修改注册代码；而对于使用 `TORCH_LIBRARY` 宏进行注册并通过 `torch.ops` 调用的算子，我们提供了同名的注册接口，用户无需修改注册代码即可完成迁移。
- **Python 接口兼容层**：对于 Python 端自定义算子封装部分，会不可避免地调用一些 PyTorch 内的 Python 组网 API。为此，我们正在致力于提升 Python 端 API 与 PyTorch 的兼容性，力求让用户在迁移过程中无需修改 Python 端代码。
- **Python API 代理层**：在 Python 端，即便 API 能够完全兼容，用户仍然需要将 `import torch` 替换为 `import paddle`。为此，我们提供了一个轻量级的代理层，用户只需在迁移后的代码开头添加一行 `import paddle.compat.enable_torch_proxy`，后续的 `torch` 下的模块将被重定向至 `paddle` 下的模块，从而实现无缝迁移。

通过以上几层兼容机制，用户可以在最大程度上复用现有的 PyTorch 自定义算子代码，从而大幅降低迁移成本。

此外，对于 TVM FFI 生态的自定义算子，由于我们已经对 TVM FFI 中所需的 DLPack 协议提供了最佳支持，因此用户可以直接将 TVM FFI 生态的自定义算子迁移至 PaddlePaddle 框架，无需额外修改。当然，如果相关算子库在 Python 端调用了 PyTorch 的组网 API，则仍然需要借助上述的 Python API 代理层来完成迁移。

## 迁移步骤

下面我们以一个简单的 PyTorch 自定义算子为例，介绍如何将其迁移至 PaddlePaddle 框架。相关代码见 [PFCCLab/cross-ecosystem-custom-op-example](https://github.com/PFCCLab/cross-ecosystem-custom-op-example)。

### 搭建 PyTorch 运行环境

在迁移之前，需要先确保 PyTorch 算子能够在本地正确编译和运行。这可以参考具体的仓库说明文档来完成。

本示例展示的自定义算子可以通过如下方式来成功编译和运行：

```bash
# 首先安装 PyTorch，具体命令可以参考 https://pytorch.org/get-started/locally/
pip install torch
# 克隆示例代码仓库
git clone https://github.com/PFCCLab/cross-ecosystem-custom-op-example.git
cd cross-ecosystem-custom-op-example
# 编译自定义算子
pip install . --no-build-isolation
# 运行测试脚本
python test.py
```

很好，我们已经确保该算子能够在 PyTorch 框架下正确运行，接下来将会介绍如何将其迁移至 PaddlePaddle 框架。

### 清理 PyTorch 环境

在迁移之前，建议先卸载 PyTorch 相关的包，或者新建一个干净的虚拟环境来进行迁移工作，以避免潜在的包冲突问题，并安装 PaddlePaddle 框架，具体命令可参考 [PaddlePaddle 安装指南](https://www.paddlepaddle.org.cn/install/quick)。

### 理解源码结构

当前示例代码的目录结构很简单，如下所示：

```text
.
├── csrc
│   └── muladd.cc       # 自定义算子实现
├── extension           # 自定义算子 Python package
│   └── __init__.py     # 自定义算子 Python 封装
├── pyproject.toml      # Python package 配置文件，主要描述 build backend
├── README.md
├── setup.py            # Python package 构建脚本，用于 build backend setuptools 的调用
└── test.py             # 测试脚本
```

从构建流程来看，我们主要关注的是：

- `pyproject.toml` 作为 PEP 517/518 标准的配置文件，描述了该 Python package 的构建后端为 `setuptools`，以及相关的元信息。

    ```toml
    [build-system]
    requires = [
        "setuptools",
        "torch",
    ]
    build-backend = "setuptools.build_meta"
    ```

- `setup.py` 作为 `setuptools` 的构建脚本，主要负责调用 `torch.utils.cpp_extension` 模块来编译 C++ 源码并生成可供 Python 调用的扩展模块。

    ```python
    from setuptools import setup, find_packages
    from torch.utils import cpp_extension

    setup(
        name="extension",
        packages=find_packages(include=['extension']),
        ext_modules=[
            cpp_extension.CUDAExtension(
                name="extension_cpp",
                sources=["csrc/muladd.cc"],
            )
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
    ```

- `csrc/muladd.cc` 作为自定义算子的核心实现文件，包含了算子的具体逻辑和注册代码，我们往往可以分为三部分：
  - 框架无关的算子逻辑实现部分，这部分逻辑并不使用 PyTorch 的 API，仅仅使用 C++/CUDA 标准库来实现。

    ```cpp
    template<typename T>
    void muladd_cpu_impl(const T* a_ptr, const T* b_ptr, T c, T* result_ptr, int64_t numel) {
        for (int64_t i = 0; i < numel; i++) {
            result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
        }
    }
    ```
  - PyTorch C++ API 相关的部分，这部分代码会使用 `at::Tensor` 等 PyTorch C++ API 来进行张量操作和内存管理。
    ```cpp
    at::Tensor muladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
        TORCH_CHECK(a.sizes() == b.sizes());
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_CHECK(b.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
        at::Tensor a_contig = a.contiguous();
        at::Tensor b_contig = b.contiguous();
        at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
        const float* a_ptr = a_contig.data_ptr<float>();
        const float* b_ptr = b_contig.data_ptr<float>();
        float* result_ptr = result.data_ptr<float>();
        muladd_cpu_impl<float>(a_ptr, b_ptr, static_cast<float>(c), result_ptr, result.numel());
        return result;
    }
    ```
  - 算子注册部分，这部分代码会使用 PyTorch 提供的注册宏（如 `TORCH_LIBRARY`）来完成算子的注册工作。

    ```cpp
    extern "C" {
        /* Creates a dummy empty _C module that can be imported from Python.
        The import from Python will load the .so consisting of this file
        in this extension, so that the TORCH_LIBRARY static initializers
        below are run. */
        PyObject* PyInit_extension_cpp(void) {
            static struct PyModuleDef module_def = {
                PyModuleDef_HEAD_INIT,
                "extension_cpp", /* name of module */
                NULL, /* module documentation, may be NULL */
                -1,   /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
                NULL, /* methods */
            };
            return PyModule_Create(&module_def);
        }
    }

    TORCH_LIBRARY(extension_cpp, m) {
        m.def("muladd_cpp(Tensor a, Tensor b, float c) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
        m.impl("muladd_cpp", &muladd_cpu);
    }
    ```


从执行流程来看，在 Python 端调用该自定义算子时，主要经历了如下几个步骤：

- 在 `test.py` 中导入自定义算子 Python package `extension`。
- 在 `extension/__init__.py` 中通过 `torch.ops.extension_cpp.muladd_cpp` 来调用 C++ 扩展模块中的自定义算子，从而调用到上面注册的 `muladd_cpp` 算子。

    ```python
    def muladd(a: torch.Tensor, b: torch.Tensor, c: float) -> torch.Tensor:
        return torch.ops.extension_cpp.muladd_cpp(a, b, c)
    ```

### 调整构建脚本，使用 PaddlePaddle 编译自定义算子

由于原本的构建脚本是基于 PyTorch 的 `torch.utils.cpp_extension` 模块来完成编译的，因此我们需要将其替换为 PaddlePaddle 提供的自定义算子编译方式。

由于我们提供了 `paddle.compat.enable_torch_proxy()` 代理层来兼容 PyTorch 的 C++ API，因此我们可以使用该 API 实现 torch API 的一键兼容调用。

```diff
+import paddle
+paddle.compat.enable_torch_proxy()  # Enable torch proxy globally

from setuptools import setup, find_packages
# 如下的 torch extension 已经被 PaddlePaddle 的同等功能替代（即 paddle.utils.cpp_extension）
# 下面的代码完全不需要修改即可运行
from torch.utils import cpp_extension

setup(
    name="extension",
    packages=find_packages(include=['extension']),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="extension_cpp",
            sources=["csrc/muladd.cc"],
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
```

对于本示例来说，仅仅需要在 `setup.py` 中添加上述两行代码即可完成迁移工作，其他代码均无需修改。但是自定义算子代码库一般各式各样，可能还需要根据实际情况进行一些调整，关于更多细节请参考 [`paddle.utils.cpp_extension.setup` 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/setup_cn.html)。

### 尝试编译并修复

完成构建脚本的修改后，即可尝试编译自定义算子：

```bash
pip install . --no-build-isolation
```

由于我们提供了 PyTorch C++ API 兼容层，因此理想情况下大多数用户的自定义算子代码都可以直接通过编译而无需修改。

PyTorch C++ API 兼容层本质上是以 PyTorch C++ API 作为用户调用接口，并在底层映射至 PaddlePaddle C++ API 来实现的。以 `at::Tensor` 为例，你所调用的 `at::Tensor` 实际上是一个代理类，该类内部持有一个 `paddle::Tensor` 对象，并将所有对 `at::Tensor` 的操作映射为对 `paddle::Tensor` 的操作。

```cpp
// paddle/phi/api/include/compat/ATen/core/TensorBody.h
namespace at {
using PaddleTensor = paddle::Tensor;

class Tensor : public TensorBase {
 public:
  Tensor() = default;
  Tensor(const PaddleTensor& tensor) : TensorBase(tensor){};  // NOLINT
  Tensor(const Tensor& tensor) = default;
  Tensor(Tensor&& tensor) = default;

  void* data_ptr() const { return const_cast<void*>(tensor_.data()); }
  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(tensor_.data<T>());
  }

  c10::IntArrayRef sizes() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.dims());
  }

  int64_t numel() const { return tensor_.numel(); }

  c10::ScalarType dtype() const {  // Should we use `TypeMeta` here?
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  c10::Device device() const { return c10::Device(tensor_.place()); }

  int64_t dim() const { return tensor_.dims().size(); }
  int64_t ndimension() const { return dim(); }

  Tensor& fill_(const at::Scalar& value) const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), value);
    return const_cast<at::Tensor&>(*this);
  }

  Tensor& zero_() const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), 0.0);
    return const_cast<at::Tensor&>(*this);
  }

  PaddleTensor _PD_GetInner() const { return tensor_; }
  PaddleTensor& _PD_GetInner() { return tensor_; }
};

}  // namespace at
namespace torch {
using at::Tensor;
}  // namespace torch
```

完整的兼容层代码见 [`paddle/phi/api/include/compat`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi/api/include/compat)，我们提供了与 PyTorch C++ API 相同的头文件结构和命名空间，只需按原有方式调用即可。

不过目前兼容层还在持续完善中，部分常见 API 尚未覆盖到，此时就会出现编译错误，你可以根据编译错误提示来定位并修复相关代码。

以 `torch::empty` 为例，假设算子库中使用了该 API，但 Paddle 没有提供该 API 的兼容实现，就会出现编译错误：

```text
/workspace/cross-ecosystem-custom-op-example/csrc/muladd.cc: In function ‘at::Tensor muladd_cpu(at::Tensor, const at::Tensor&, double)’:
/workspace/cross-ecosystem-custom-op-example/csrc/muladd.cc:54:30: error: ‘empty’ is not a member of ‘torch’
   54 |   at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
      |                              ^~~~~
```

此时我们可以选择将 PyTorch 的 structs 转换为 Paddle 的 structs，并用 PaddlePaddle 提供的等效 API 来实现该功能：

即将下面的代码：

```cpp
// PyTorch 原代码
at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
```

我们可以将其替换为：

```cpp
// 替换为 PaddlePaddle 等效实现
auto paddle_size = a_contig.sizes()._PD_ToPaddleIntArray();  // 将 PyTorch IntArrayRef 转为 Paddle IntArray
auto paddle_dtype = compat::_PD_AtenScalarTypeToPhiDataType(a_contig.dtype());  // 将 PyTorch ScalarType 转为 Paddle DataType
auto paddle_place = a_contig.options()._PD_GetPlace();  // 将 PyTorch Device 转为 Paddle Place
auto paddle_result = paddle::experimental::empty(paddle_size, paddle_dtype, paddle_place);  // 调用 PaddlePaddle 的 empty API
at::Tensor result(paddle_result);  // 将 Paddle Tensor 包装为 PyTorch Tensor
```

更多 PaddlePaddle C++ API 的使用方式可参考 [PaddlePaddle C++ 自定义算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html)。通过这种方式，你可以逐步修复编译错误，直至自定义算子能够成功编译通过。

### 运行测试并修复

完成编译后，即可运行测试脚本来验证自定义算子的正确性，由于原本的测试脚本是基于 PyTorch 框架来实现的，因此我们需要改写测试脚本以适配 PaddlePaddle 框架。

```python
import paddle
paddle.compat.enable_torch_proxy(scope={"extension"})  # 仅启用 extension 包的 torch 代理
import extension

x = paddle.tensor([1.0, 2.0, 3.0])
y = paddle.tensor([4.0, 5.0, 6.0])
z = 2.0
result = extension.muladd(x, y, z)
print(result)  # Expected output: tensor([ 6., 12., 20.])
```

由于 `extension` 包中仍然使用了 `torch` 模块下的 Python API，因此我们需要启用 `torch` 代理来确保这些 API 能够正确映射至 PaddlePaddle 框架。为了避免对其他代码产生影响，我们可以通过 `scope` 参数来限定代理的作用范围。

当然，与 C++ 端类似，Python 端的兼容层也还在持续完善中，部分常见 API 尚未覆盖到，如果遇到此类错误，你可以尝试参考 [PaddlePaddle Python API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html) 和 [PyTorch 最新 release 与 Paddle develop API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)来寻找等效的 PaddlePaddle API 并进行替换，直到运行时不再报错且结果正确为止。

至此，一个 PyTorch 自定义算子就成功迁移至 PaddlePaddle 框架了！

### 总结

通过上述步骤，我们介绍了如何将一个简单的 PyTorch 自定义算子迁移至 PaddlePaddle 框架。总体来说，迁移工作主要包括以下几个方面：

- 调整构建脚本，使用 PaddlePaddle 提供的自定义算子编译方式来替换原有的 PyTorch 构建方式。
- 修复 C++ 端的编译错误，主要是由于部分 PyTorch C++ API 尚未覆盖到，需要借助 PaddlePaddle C++ API 来实现等效功能。
- 改写 Python 端的测试脚本，借助 torch proxy 代理层一键替换 PyTorch Python API，并根据实际情况进行部分 API 替换。

目前无论是 C++ 端还是 Python 端的兼容层都还在持续完善中，未来我们会不断补充更多常用 API 的兼容实现，从而进一步降低用户的迁移成本。同时我们也非常欢迎社区用户参与到兼容层的建设中来，共同推动跨生态自定义算子的互通与发展！
