.. _cn_api_paddle_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sqrt(x, name=None)




计算输入的算数平方根。

.. math:: out=\sqrt x=x^{1/2}

.. note::
    请确保输入中的数值是非负数。

参数
::::::::::::


    - **x** (Tensor) - 支持任意维度的 Tensor。数据类型为 float32，float64 或 float16。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回类型为 Tensor，数据类型同输入一致。

代码示例
::::::::::::

COPY-FROM: paddle.sqrt
