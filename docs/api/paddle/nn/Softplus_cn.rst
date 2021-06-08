.. _cn_api_nn_Softplus:

Softplus
-------------------------------
.. py:class:: paddle.nn.Softplus(beta=1, threshold=20, name=None)

Softplus激活层

.. math::

    Softplus(x) = \frac{1}{beta} * \log(1 + e^{beta * x}) \\
    \text{为了保证数值稳定性, 当}\,beta * x > threshold\,\text{时，函数转变为线性函数x}.

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - beta (float, 可选) - Softplus激活计算公式中的beta值。默认值为1。
    - threshold (float, 可选) - Softplus激活计算公式中的threshold值。默认值为20。
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

    x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
    m = paddle.nn.Softplus()
    out = m(x) # [0.513015, 0.598139, 0.744397, 0.854355]
