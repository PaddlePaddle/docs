.. _cn_api_nn_Softshrink:

Softshrink
-------------------------------
.. py:class:: paddle.nn.Softshrink(threshold=0.5, name=None)

Softshrink激活层

.. math::

    Softshrink(x)= \begin{cases}
                    x - threshold, \text{if } x > threshold \\
                    x + threshold, \text{if } x < -threshold \\
                    0,  \text{otherwise}
                    \end{cases}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - threshold (float, 可选) - Softshrink激活计算公式中的threshold值，必须大于等于零。默认值为0.5。
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

    x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
    m = paddle.nn.Softshrink()
    out = m(x) # [-0.4, 0, 0, 0.3]
