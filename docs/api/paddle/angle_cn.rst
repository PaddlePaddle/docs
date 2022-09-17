.. _cn_api_paddle_angle:

angle
-------------------------------

.. py:function:: paddle.angle(x, name=None)


逐元素计算复数的相位角。对于非负实数，相位角为 0，而对于负实数，相位角为 :math:`\pi`。

.. math::

    angle(x) = arctan2(x.imag, x.real)

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：complex64, complex128 或 float32, float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出实数 Tensor，与 ``x`` 的数值精度一致。

代码示例
:::::::::

COPY-FROM: paddle.angle
