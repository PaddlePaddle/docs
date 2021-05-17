.. _cn_api_nn_Hardswish:

Hardswish
-------------------------------

.. py:function:: paddle.nn.Hardswish(name=None)

Hardswish激活函数。在MobileNetV3架构中被提出，相较于swish函数，具有数值稳定性好，计算速度快等优点，具体原理请参考: https://arxiv.org/pdf/1905.02244.pdf

.. math::

    hardswish(x)=
        \left\{
        \begin{aligned}
        &0, & & \text{if } x \leq -3 \\
        &x, & & \text{if } x \geq 3 \\
        &\frac{x(x+3)}{6}, & & \text{otherwise}
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-4., 5., 1.])
    m = paddle.nn.Hardswish()
    out = m(x) # [0., 5., 0.666667]
