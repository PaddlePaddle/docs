.. _cn_api_nn_cn_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.nn.functional.softshrink(x, threshold=0.5, name=None)

softshrink激活层

.. math::

    softshrink(x)= \begin{cases}
                    x - threshold, \text{if } x > threshold \\
                    x + threshold, \text{if } x < -threshold \\
                    0,  \text{otherwise}
                    \end{cases}

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - threshold (float, 可选) - softshrink激活计算公式中的threshold值，必须大于等于零。默认值为0.5。
 - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np

    x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
    out = F.softshrink(x) # [-0.4, 0, 0, 0.3]
