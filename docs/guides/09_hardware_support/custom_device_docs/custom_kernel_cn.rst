#############
# 自定义 Kernel 教程
#############

内核函数（简称Kernel）对应算子的具体实现，飞桨框架针对通过自定义Runtime机制注册的外部硬件，提供了配套的自定义Kernel机制，以实现独立于框架的Kernel编码、注册、编译和自动加载使用。
自定义Kernel基于飞桨对外发布的函数式Kernel声明、对外开放的C++ API和注册宏实现。


- `Kernel函数声明 <./custom_kernel_docs/kernel_declare_cn.html>`_ : 介绍飞桨发布的函数式Kernel声明。
- `Kernel实现API <./custom_kernel_docs/cpp_api_cn.html>`_ : 介绍自定义Kernel函数体实现所需的C++ API。
- `Kernel注册接口 <./custom_kernel_docs/register_api_cn.html>`_ : 介绍自定义Kernel注册接口。
- `自定义Kernel举例 <./custom_kernel_docs/abs_example_cn.html>`_ : 通过示例介绍自定义Kernel完整流程。


..  toctree::
    :hidden:

    custom_kernel_docs/kernel_declare_cn.md
    custom_kernel_docs/cpp_api_cn.rst
    custom_kernel_docs/register_api_cn.md
    custom_kernel_docs/abs_example_cn.md