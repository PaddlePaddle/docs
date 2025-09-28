.. _cn_api_paddle_stanh:

stanh
-------------------------------

.. py:function:: paddle.stanh(x, scale_a=0.67, scale_b=1.7159, name=None)

stanh 激活函数

.. math::

    out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

参数
::::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **scale_a** (float，可选) - stanh 激活计算公式中的输入缩放参数 a。默认值为 0.67。
    - **scale_b** (float，可选) - stanh 激活计算公式中的输出缩放参数 b。默认值为 1.7159。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.stanh
