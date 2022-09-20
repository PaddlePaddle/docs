####################
自定义新硬件接入指南
####################

自定义硬件接入功能实现框架和硬件的解耦，提供一种插件式扩展 PaddlePaddle 硬件后端的方式。通过该功能，开发者无需为特定硬件修改 PaddlePaddle 代码，只需实现标准接口，并编译成动态链接库，则可作为插件供 PaddlePaddle 调用。降低为 PaddlePaddle 添加新硬件后端的开发难度。

自定义硬件接入功能由自定义 Runtime 与自定义 Kernel 两个主要组件构成，基于这两个组件，用户可按需完成自定义新硬件接入飞桨。

- `自定义 Runtime <./custom_runtime_cn.html>`_ : 飞桨框架自定义 Runtime 介绍
- `自定义 Kernel <./custom_kernel_cn.html>`_ : 飞桨框架自定义 Kernel 介绍
- `新硬件接入示例 <./custom_kernel_cn.html>`_ : 通过示例介绍自定义新硬件接入飞桨的步骤

..  toctree::
    :hidden:


    custom_runtime_cn.rst
    custom_kernel_cn.rst
    custom_device_example_cn.md
