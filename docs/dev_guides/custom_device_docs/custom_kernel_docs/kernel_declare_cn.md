# Kernel 函数声明

飞桨通过头文件发布函数式 Kernel 声明，框架内外一致。

编写自定义 Kernel 需基于具体的 Kernel 函数声明，头文件位于飞桨安装路径的`include/paddle/phi/kernels/`下。

Kernel 函数声明的格式如下：

```c++
template <typename T, typename Context>
void KernelNameKernel(const Context& dev_ctx,
                      InputTensor(s),
                      Attribute(s),
                      OutTensor(s));
```

约定：

1. 模板参数：固定写法，第一个模板参数为数据类型`T`，第二个模板参数为设备上下文`Context`。
2. 函数返回：固定为`void`。
3. 函数命名：Kernel 名称+Kernel 后缀，驼峰式命名，如`SoftmaxKernel`。
4. 函数参数：依次为设备上下文参数，输入 Tensor 参数（InputTensor），属性参数（Attribute）和输出 Tensor 参数（OutTensor）。其中：
- 设备上下文参数：固定为`const Context&`类型；
    - 自定义 Kernel 对应`CustomContext`类型，请参照[custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h)
- InputTensor：数量>=0，支持的类型包括：
    - `const DenseTensor&` 请参照[dense_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/dense_tensor.h)
    - `const SelectedRows&` 请参照[selected_rows.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/selected_rows.h)
    - `const SparseCooTensor&` 请参照[sparse_coo_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_coo_tensor.h)
    - `const SparseCsrTensor&` 请参照[sparse_csr_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_csr_tensor.h)
    - `const std::vector<DenseTensor*>&`
    - `const std::vector<SparseCooTensor*>&`
    - `const std::vector<SparseCsrTensor*>&`
- Attribute：数量>=0，支持的类型包括：
    - `bool`
    - `float`
    - `double`
    - `int`
    - `int64_t`
    - `phi::dtype::float16` 请参照[float16.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float16.h)
    - `const Scalar&` 请参照[scalar.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/scalar.h)
    - `DataType` 请参照[data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
    - `DataLayout` 请参照[layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
    - `Place` 请参照[place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
    - `const std::vector<int64_t>&`
    - `const ScalarArray&` 请参照[int_array.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/int_array.h)
    - `const std::vector<int>&`
    - `const std::string&`
    - `const std::vector<bool>&`
    - `const std::vector<float>&`
    - `const std::vector<double>&`
    - `const std::vector<std::string>&`
- OutTensor：数量>0，支持的类型包括：
    - `DenseTensor*`
    - `SelectedRows*`
    - `SparseCooTensor*`
    - `SparseCsrTensor*`
    - `std::vector<DenseTensor*>`
    - `std::vector<SparseCooTensor*>`
    - `std::vector<SparseCsrTensor*>`

示例，如`softmax`的 Kernel 函数位于`softmax_kernel.h`中，具体如下：

```c++
// Softmax 内核函数
// 模板参数： T - 数据类型
//          Context - 设备上下文
// 参数： dev_ctx - Context 对象
//       x - DenseTensor 对象
//       axis - int 类型
//       dtype - DataType 类型
//       out - DenseTensor 指针
// 返回： None
template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DataType dtype,
                   DenseTensor* out);
```

> 注意：
> 1. Kernel 函数声明是自定义 Kernel 能够被注册和框架调用的基础，由框架发布，需要严格遵守
> 2. Kernel 函数声明与头文件可能不完全对应，可以按照函数命名约定等查找所需 Kernel 函数声明
