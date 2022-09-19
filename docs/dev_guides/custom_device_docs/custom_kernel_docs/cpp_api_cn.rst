#############
Kernel 实现接口
#############

自定义 Kernel 函数体的实现主要依赖两部分：1.飞桨发布的 API：如设备上下文 API、Tensor 相关 API 和异常处理 API 等； 2.硬件封装库的 API：根据具体硬件封装库使用。其中飞桨发布的 C++ API 已通过头文件方式发布。


- `Context API <./context_api_cn.html>`_ : 介绍设备上下文相关 C++ API。
- `Tensor API <./tensor_api_cn.html>`_ : 介绍 Tensor 相关 C++ API。
- `Exception API <./exception_api_cn.html>`_ : 介绍异常处理相关 C++ API。


注：飞桨发布了丰富的 C++ API，此处重点介绍三类 API 并在相应页面列举相关联的类和文件供开发者参考查阅。

..  toctree::
    :hidden:

    context_api_cn.md
    tensor_api_cn.md
    exception_api_cn.md
