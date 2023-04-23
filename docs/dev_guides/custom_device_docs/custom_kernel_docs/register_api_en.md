# Kernel Registration API

The registration macro of PaddlePaddle helps to register the custom kernel，which can be called by the PaddlePaddle framework.

The registration macro should be put in a global space.

For the basic format of the registration macro, please refer to [kernel_registry.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/kernel_registry.h)

```c++
/** PD_REGISTER_PLUGIN_KERNEL
 *
 * Used to register kernels for plug-in backends.
 * Support user-defined backend such as 'Ascend910'.
 */
PD_REGISTER_PLUGIN_KERNEL(kernel_name, backend, layout, meta_kernel_fn, ...)) {}
```

Explanation：

- Name of the macro：`PD_REGISTER_PLUGIN_KERNEL`
- First parameter：kernel_name，which is the same both inside and outside. You can refer to registration names of the same kernel functions of CPU, such as `softmax`.
- Second parameter：backend，which can be customized. But its name must be the same as that of the custom runtime, such as `Ascend910`.
- Third parameter：layout，the enumeration of `DataLayout`. For the setting, please refer to [layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
- Fourth parameter：meta_kernel_fn，the name of a kernel function. Here, the template parameter is not included, such as `my_namespace::SoftmaxKernel`.
- Variable-length data type parameter: includes basic C++ data types or types defined by PaddlePaddle like `phi::dtype::float16`、`phi::dtype::bfloat16`、`phi::dtype::complex`. You can refer to [data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
- End：the function body. You can set the kernel if necessary. If not, keep `{}`.

>Explanation: the declaration corresponding to the end function body：
>```c++
>// Kernel Parameter Definition
>// Parameter： kernel_key - KernelKey object
>//       kernel - Kernel pointer
>// Return： None
>void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(
>      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
>```
> You can use the parameters `kernel_key` and `kernel` in the function body，and customize the kernel in its registration.

Take the registration of the CustomCPU backend kernel of `softmax` as an example:

```c++
// The registration of the CustomCPU backend kernel of `softmax`
// Global naming space
// Parameter： softmax - Kernel name
//       CustomCPU - Backend name
//       ALL_LAYOUT - Storage layout
//       custom_cpu::SoftmaxKernel - Kernel function name
//       float - name of the data type
//       double - name of the data type
//       phi::dtype::float16 - name of the data type
PD_REGISTER_PLUGIN_KERNEL(softmax,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::SoftmaxKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
```

> Note：
> 1. When the backend is registered through the custom runtime, the backend parameter must be the same as its name.
> 2. Except the requirement of the end function body of the registration macro，keep the empty function body. You can refer to other backends within the PaddlePaddle framework.
