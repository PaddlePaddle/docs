.. _cn_api_nn_cn_swish:

swish
-------------------------------

.. py:function:: paddle.nn.functional.swish(x, name=None)

swish激活层。计算公式如下：

.. math::

    swish(x) = \frac{x}{1 + e^{-x}}

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

    x = paddle.to_tensor(np.array([-2., 0., 1.]))
    out = F.swish(x) # [-0.238406, 0., 0.731059]
