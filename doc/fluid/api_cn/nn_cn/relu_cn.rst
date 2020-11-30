.. _cn_api_nn_cn_relu:

relu
-------------------------------

.. py:function:: paddle.nn.functional.relu(x, name=None)

relu激活层（Rectified Linear Unit）。计算公式如下：

.. math::

    relu(x) = max(0, x)

其中，:math:`x` 为输入的 Tensor


参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
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

    paddle.disable_static()

    x = paddle.to_tensor(np.array([-2, 0, 1]).astype('float32'))
    out = F.relu(x) # [0., 0., 1.]
