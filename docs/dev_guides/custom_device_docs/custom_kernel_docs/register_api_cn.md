# Kernel 注册接口

自定义 Kernel 通过飞桨框架提供的注册宏进行注册，以便飞桨框架调用。

注册宏的位置需要放置在全局空间下。

注册宏的基本形式如下：具体实现请参照[kernel_registry.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/kernel_registry.h)

```c++
/** PD_REGISTER_PLUGIN_KERNEL
 *
 * Used to register kernels for plug-in backends.
 * Support user-defined backend such as 'Ascend910'.
 */
PD_REGISTER_PLUGIN_KERNEL(kernel_name, backend, layout, meta_kernel_fn, ...)) {}
```

说明：

- 注册宏名称：固定为`PD_REGISTER_PLUGIN_KERNEL`
- 第一个参数：kernel_name，即 Kernel 名称，飞桨内外一致，请参照 CPU 相同 Kernel 函数注册名称，如`softmax`
- 第二个参数：backend，即后端名称，可自定义，但须与自定义 Runtime 设定的名称一致，如`Ascend910`
- 第三个参数：layout，即内存布局，为`DataLayout`类型的枚举，按需设定，请参照[layout.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/layout.h)
- 第四个参数：meta_kernel_fn，即 Kernel 函数名，注意此处不加模板参数，如`my_namespace::SoftmaxKernel`
- 不定长数据类型参数：C++的基础数据类型或飞桨定义的`phi::dtype::float16`、`phi::dtype::bfloat16`、`phi::dtype::complex`等类型，请参照[data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
- 末尾：固定为函数体，其中可按需对 Kernel 进行必要设置，如果没有，保留`{}`。

>说明：末尾函数体对应的函数声明如下：
>```c++
>// Kernel 参数定义
>// 参数： kernel_key - KernelKey 对象
>//       kernel - Kernel 指针
>// 返回： None
>void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(
>      const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
>```
>即函数体中可使用参数`kernel_key`与`kernel`，在 Kernel 注册时对 Kernel 进行个性化调整。

示例，如`softmax`的 CustomCPU 后端 Kernel 注册如下：

```c++
// Softmax 的 CustomCPU 后端 Kernel 注册
// 全局命名空间
// 参数： softmax - Kernel 名称
//       CustomCPU - 后端名称
//       ALL_LAYOUT - 内存布局
//       custom_cpu::SoftmaxKernel - Kernel 函数名
//       float - 数据类型名
//       double - 数据类型名
//       phi::dtype::float16 - 数据类型名
PD_REGISTER_PLUGIN_KERNEL(softmax,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::SoftmaxKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
```

> 注意：
> 1. 对于通过自定义 Runtime 接入的后端，backend 参数须与之名称保持一致
> 2. 注册宏末尾函数体中除非有明确需要，否则保留空函数体即可，请参照飞桨框架内其它后端的使用
