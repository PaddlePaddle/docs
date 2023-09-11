.. _cn_api_paddle_jit_enable_to_static:

enable_to_static
-------------------------------

.. py:function:: paddle.jit.enable_to_static(enable_to_static_bool)

全局启用或禁用从动态图到静态图的转换。


参数
::::::::::::

    - **enable_to_static_bool** (bool) - 启用或禁用动转静。为 `True` 时启用动转静, 为 `False` 时关闭动转静。


代码示例
::::::::::::

COPY-FROM: paddle.jit.enable_to_static
