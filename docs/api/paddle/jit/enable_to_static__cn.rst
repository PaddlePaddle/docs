.. _cn_api_paddle_jit_enable_to_static:

enable_to_static
-------------------------------

.. py:function:: paddle.jit.enable_to_static(enable_to_static_bool)
通过 ProgramTranslator 全局地启用或禁用从命令图到静态图的转换。

参数
:::::::::

**enable_to_static_bool** (bool) – True 或 False 可启用或禁用转换为静态。

返回
:::::::::

None

代码示例
::::::::::::

COPY-FROM: paddle.jit.enable_to_static
