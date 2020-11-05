.. _cn_api_nn_cn_elu:

elu
-------------------------------

.. py:function:: paddle.nn.functional.elu(x, alpha=1.0, name=None)

elu激活层（ELU Activation Operator）

根据 `Exponential Linear Units <https://arxiv.org/abs/1511.07289>`_ 对输入Tensor中每个元素应用以下计算。

.. math::

    elu(x) = max(0, x) + min(0, \alpha * (e^{x} − 1))

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - alpha (float, 可选) - elu的alpha值，默认值为1.0。
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

    x = paddle.to_tensor(np.array([[-1,6],[1,15.6]]))
    out = F.elu(x, alpha=0.2) 
    # [[-0.12642411  6.        ]
    #  [ 1.          15.6      ]]


