####################
自定义硬件接入
####################

自定义硬件接入功能实现框架和硬件的解耦，提供一种插件式扩展 PaddlePaddle 硬件后端的方式。通过该功能，开发者无需为特定硬件修改 PaddlePaddle 代码，只需实现标准接口，并编译成动态链接库，则可作为插件供 PaddlePaddle 调用。降低为 PaddlePaddle 添加新硬件后端的开发难度。

自定义硬件接入功能由两个主要组件构成：

- `自定义 Runtime <./custom_runtime_cn.html>`_ 
- 自定义 Kernel


..  toctree::
    :hidden:

    custom_runtime_cn.rst
    custom_device_example_cn.md
