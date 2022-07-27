.. _cn_api_paddle_tensor_math_log2:

log2
-------------------------------

.. py:function:: paddle.log2(x, name=None)





Log2激活函数（计算底为2的对数）

.. math::
                  \\Out=log_2x\\


参数
:::::::::
  - **x** (Tensor) – 该OP的输入为Tensor。数据类型为float32，float64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor,Log2算子底为2对数输出，数据类型与输入一致。


代码示例
:::::::::

COPY-FROM: paddle.log2