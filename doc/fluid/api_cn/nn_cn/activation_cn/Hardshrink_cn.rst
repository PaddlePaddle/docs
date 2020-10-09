.. _cn_api_nn_Hardshrink:

Hardshrink
-------------------------------
.. py:class:: paddle.nn.Hardshrink(threshold=0.5, name=None)

Hardshrink激活层

.. math::

    Hardshrink(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x > threshold \\
        &x, & & if \ x < -threshold \\
        &0, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - threshold (float, 可选) - Hardshrink激活计算公式中的threshold值。默认值为0.5。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    x = paddle.to_tensor([-1, 0.3, 2.5])
    m = paddle.nn.Hardshrink()
    out = m(x) # [-1., 0., 2.5]
