# Kernel Function Declaration

PaddlePaddle has released the kernel declaration through the header file, and the framework is uniform both inside and outside.

Custom kernel editing should be based on a specific kernel function declaration. The header file is under `include/paddle/phi/kernels/`.

The format of the declaration is as follows：

```c++
template <typename T, typename Context>
void KernelNameKernel(const Context& dev_ctx,
                      InputTensor(s),
                      Attribute(s),
                      OutTensor(s));
```

Agreement：

1. Template Parameter：It is fixed in format. The data type of the first parameter is `T`，and that of the second is `Context`.
2. Return：`void` is the pattern.
3. Naming：Camel case: kernel name + "Kernel"，such as `SoftmaxKernel`
4. Parameter：Context parameter, InputTensor，Attribute，and OutTensor, all arranged in order:
- Context Parameter：It belongs to `const Context&`.
    - `CustomContext` corresponding with the custom kernel. You can refer to [custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h)
- InputTensor：Number >=0，and the types include：
    - `const DenseTensor&` Please refer to [dense_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/dense_tensor.h)
    - `const SelectedRows&` Please refer to [selected_rows.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/selected_rows.h)
    - `const SparseCooTensor&` Please refer to [sparse_coo_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_coo_tensor.h)
    - `const SparseCsrTensor&` Please refer to [sparse_csr_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/sparse_csr_tensor.h)
    - `const std::vector<DenseTensor*>&`
    - `const std::vector<SparseCooTensor*>&`
    - `const std::vector<SparseCsrTensor*>&`
- Attribute：Number >=0，and the types include：
    - `bool`
    - `float`
    - `double`
    - `int`
    - `int64_t`
    - `phi::dtype::float16` Please refer to [float16.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float16.h)
    - `const Scalar&` Please refer to [scalar.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/scalar.h)
    - `DataType` Please refer to [data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
    - `DataLayout` Please refer to [layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
    - `Place` Please refer to [place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
    - `const std::vector<int64_t>&`
    - `const ScalarArray&` Please refer to [scalar_array.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/scalar_array.h)
    - `const std::vector<int>&`
    - `const std::string&`
    - `const std::vector<bool>&`
    - `const std::vector<float>&`
    - `const std::vector<double>&`
    - `const std::vector<std::string>&`
- OutTensor：Number >0，and the types include：
    - `DenseTensor*`
    - `SelectedRows*`
    - `SparseCooTensor*`
    - `SparseCsrTensor*`
    - `std::vector<DenseTensor*>`
    - `std::vector<SparseCooTensor*>`
    - `std::vector<SparseCsrTensor*>`

For example，when the kernel function of `softmax` is in `softmax_kernel.h`:

```c++
// Softmax
// Template Parameter： T - data type
//          Context - the device context
// Parameter： dev_ctx - object of the Context
//       x - DenseTensor object
//       axis - int type
//       dtype - DataType type
//       out - DenseTensor pointer
// Return： None
template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DataType dtype,
                   DenseTensor* out);
```

> Note：
> 1. The kernel function declaration is the basis of the registration and the framework invocation of the custom kernel. It is released by the framework and required to be observed.
> 2. The kernel function declaration cannot perfectly match the header file. You can find the declaration you need by searching the name of the function.
