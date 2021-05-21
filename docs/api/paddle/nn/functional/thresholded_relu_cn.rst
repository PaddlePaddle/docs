.. _cn_api_nn_cn_thresholded_relu:

thresholded_relu
-------------------------------

.. py:function:: paddle.nn.functional.thresholded_relu(x, threshold=1.0, name=None)

thresholded relu激活层。计算公式如下：

.. math::

    thresholded\_relu(x) = \begin{cases}
                            x, \text{if } x > threshold \\
                            0, \text{otherwise}
                           \end{cases}

其中，:math:`x` 为输入的 Tensor


参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - threshold (float, 可选) - thresholded_relu激活计算公式中的threshold值。默认值为1.0。
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

    x = paddle.to_tensor(np.array([2., 0., 1.]))
    out = F.thresholded_relu(x) # [2., 0., 0.]
