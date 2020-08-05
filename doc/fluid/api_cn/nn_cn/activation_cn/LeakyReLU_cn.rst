.. _cn_api_nn_LeakyReLU:

LeakyReLU
-------------------------------
.. py:class:: paddle.nn.LeakyReLU(alpha=0.01, name=None)

ReLU (Rectified Linear Unit）激活层

.. math::

        \\Out = max(x, alpha*x)\\

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - alpha (float，可选) - :math:`x < 0` 时的斜率。默认值为0.01。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    无

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()

    lrelu = nn.LeakyReLU()
    x = paddle.imperative.to_variable(np.array([-2, 0, 1]))
    res = lrelu(x)  # [-0.02, 0, 1]
