.. _cn_api_tensor_tanh:

tanh
-------------------------------

.. py:function:: paddle.tanh(x, name=None)


tanh 激活函数

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数
:::::::::


    - **x** (Tensor) - Tanh算子的输入，多维Tensor，数据类型为 float16，float32或float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tanh的输出Tensor，和输入有着相同类型和shape。


代码示例
:::::::::

COPY-FROM: paddle.tanh
