.. _cn_api_nn_cn_tanhshrink:

tanhshrink
-------------------------------

.. py:function:: paddle.nn.functional.tanhshrink(x, name=None)

tanhshrink激活层

.. math::

    tanhshrink(x) = x - tanh(x)

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
    out = F.tanhshrink(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]
