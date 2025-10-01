.. _cn_api_paddle_isreal:

isreal
-------------------------------

.. py:function:: paddle.isreal(x, name=None)


判断输入 tensor 的每一个值是否为实数类型(非 complex64 或者 complex128)。

参数
:::::::::
   - **x** (Tensor) - 输入 Tensor。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，每个元素是一个 bool 值，表示输入 `x` 的每个元素是否为实数类型。


代码示例
:::::::::

COPY-FROM: paddle.isreal
