.. _cn_api_fluid_layers_stanh:

stanh
-------------------------------

.. py:function:: paddle.fluid.layers.stanh(x, scale_a=0.67, scale_b=1.7159, name=None)

stanh 激活函数

.. math::

    out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - scale_a (float, 可选) - stanh激活计算公式中的输入缩放参数a。默认值为0.67。
 - scale_b (float, 可选) - stanh激活计算公式中的输出缩放参数b。默认值为1.7159。
 - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
    out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]
