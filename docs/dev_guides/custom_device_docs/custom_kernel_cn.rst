####################
自定义 Kernel
####################

内核函数（简称 Kernel）对应算子的具体实现，飞桨框架针对通过自定义 Runtime 机制注册的外部硬件，提供了配套的自定义 Kernel 机制，以实现独立于框架的 Kernel 编码、注册、编译和自动加载使用。
自定义 Kernel 基于飞桨对外发布的函数式 Kernel 声明、对外开放的 C++ API 和注册宏实现。


- `Kernel 函数声明 <./custom_kernel_docs/kernel_declare_cn.html>`_ : 介绍飞桨发布的函数式 Kernel 声明。
- `Kernel 实现接口 <./custom_kernel_docs/cpp_api_cn.html>`_ : 介绍自定义 Kernel 函数体实现所需的 C++ API。
- `Kernel 注册接口 <./custom_kernel_docs/register_api_cn.html>`_ : 介绍自定义 Kernel 注册宏。


..  toctree::
    :hidden:

    custom_kernel_docs/kernel_declare_cn.md
    custom_kernel_docs/cpp_api_cn.rst
    custom_kernel_docs/register_api_cn.md
