.. _cn_api_fluid_layers_exp:

exp
-------------------------------

.. py:function:: paddle.exp(x, name=None)




对输入，逐元素进行以自然数e为底指数运算。

.. math::
    out = e^x

参数
::::::::::::

    - **x** (Tensor) - 该OP的输入为多维Tensor。数据类型为float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出为Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
::::::::::::

COPY-FROM: paddle.exp