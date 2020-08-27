.. _cn_api_nn_SELU:

SELU
-------------------------------
.. py:class:: paddle.nn.SELU(scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)

SELU激活层

.. math::

    SELU(x)= scale *
             \begin{cases}
               x, \text{if } x > 0 \\
               alpha * e^{x} - alpha, \text{if } x <= 0
             \end{cases}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - scale (float, 可选) - SELU激活计算公式中的scale值，必须大于1.0。默认值为1.0507009873554804934193349852946。
    - alpha (float, 可选) - SELU激活计算公式中的alpha值，必须大于等于零。默认值为1.6732632423543772848170429916717。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    x = paddle.to_tensor(np.array([[0.0, 1.0],[2.0, 3.0]]))
    m = paddle.nn.SELU()
    out = m(x) # [[0, 1.050701],[2.101402, 3.152103]]
