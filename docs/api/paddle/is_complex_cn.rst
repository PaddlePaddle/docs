.. _cn_api_paddle_is_complex:

is_complex
-------------------------------

.. py:function:: paddle.is_complex(x)

判断输入 tensor 的数据类型是否为复数类型(complex64 或者 complex128)。

.. note::
   别名支持: 参数名 ``input`` 可替代 ``x``，如 ``input=tensor_x`` 等价于 ``x=tensor_x``。

参数
:::::::::
   - **x** (Tensor) - 输入 Tensor。别名： ``input``。


返回
:::::::::
bool，如果输入 tensor 的数据类型为复数类型则为 True，反之为 False


代码示例
:::::::::

COPY-FROM: paddle.is_complex
