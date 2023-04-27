.. _cn_api_tensor_tanh:

tanh
-------------------------------

.. py:function:: paddle.tanh(x, name=None)


tanh 激活函数

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数
:::::::::


    - **x** (Tensor) - Tanh 算子的输入，多维 Tensor，数据类型为 bfloat16，float16，float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tanh 的输出 Tensor，和输入有着相同类型和 shape。


代码示例
:::::::::

COPY-FROM: paddle.tanh
