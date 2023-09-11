####################
硬件接入飞桨后端指南
####################

飞桨深度学习框架提供了多种硬件接入的方案，包括算子开发与映射、子图与整图接入、深度学习编译器后端接入以及开源神经网络格式转化四种硬件接入方案，供硬件厂商根据自身芯片架构设计与软件栈的建设成熟度进行灵活选择。

- `硬件接入飞桨后端方案介绍 <./new_device_backend_overview_cn.html>`_ : 新硬件接入飞桨框架后端的整体方案介绍
- `训练硬件 Custom Device 接入方案介绍 <./custom_device_overview_cn.html>`_ : 训练硬件建议采用可解耦插件式接入方案，本文介绍 Custom Device 整体方案。
 - `自定义 Runtime <./custom_runtime_cn.html>`_ : 飞桨框架自定义 Runtime 介绍
 - `自定义 Kernel <./custom_kernel_cn.html>`_ : 飞桨框架自定义 Kernel 介绍
 - `新硬件接入示例 <./custom_device_example_cn.html>`_ : 通过示例介绍自定义新硬件接入飞桨的步骤
- `推理硬件 NNAdapter 接入方案介绍 <https://www.paddlepaddle.org.cn/lite/v2.11/develop_guides/nnadapter.html>`_ : 推理硬件如果需接入飞桨轻量化推理引擎 Paddle Lite 后端，建议采用图接入方案，本文介绍 NNAdapter 整体方案。

..  toctree::
    :hidden:


    new_device_backend_overview_cn.rst
    custom_device_overview_cn.rst
    custom_runtime_cn.rst
    custom_kernel_cn.rst
    custom_device_example_cn.md
