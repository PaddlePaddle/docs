.. _cn_api_paddle_outer:

outer
-------------------------------

.. py:function:: paddle.outer(x, y, name=None)


计算两个 Tensor 的外积。

对于 1 维 Tensor 正常计算外积，对于大于 1 维的 Tensor 先展平为 1 维再计算外积。

.. note::
    别名支持: 参数名  ``input``  可替代  ``x`` ，参数名  ``vec2``  可替代  ``y``  ，如  ``outer(input=tensor_x, vec2=tensor_y, ...)``  等价于  ``outer(x=tensor_x, y=tensor_y, ...)``  。

参数
:::::::::

    - **x** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor。别名：  ``input`` 。
    - **y** (Tensor) - 一个 N 维 Tensor 或者标量 Tensor。别名：  ``vec2`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor, x、y 的外积结果，Tensor shape 为 [x.size, y.size]。

代码示例：
::::::::::

COPY-FROM: paddle.outer
