.. _cn_api_nn_cn_softsign:

softsign
-------------------------------

.. py:function:: paddle.nn.functional.softsign(x, name=None)

softsign激活层

.. math::

    softsign(x) = \frac{x}{1 + |x|}

其中，:math:`x` 为输入的 Tensor

参数:
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

    x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
    out = F.softsign(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]
