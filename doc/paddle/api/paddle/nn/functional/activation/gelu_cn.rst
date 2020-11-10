.. _cn_api_nn_cn_gelu:

gelu
-------------------------------

.. py:function:: paddle.nn.functional.gelu(x, approximate=False, name=None)

gelu激活层（GELU Activation Operator）

逐元素计算 gelu激活函数。更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ 。

如果使用近似计算：

.. math::
    gelu(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    gelu(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - approximate (bool, 可选) - 是否使用近似计算，默认值为 False，表示不使用近似计算。
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

    x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))
    out1 = F.gelu(x) # [-0.158655 0.345731 0.841345 1.39979]
    out2 = F.gelu(x, True) # [-0.158808 0.345714 0.841192 1.39957]

