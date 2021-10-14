.. _cn_api_nn_cn_celu:

celu
-------------------------------

.. py:function:: paddle.nn.functional.celu(x, alpha=1.0, name=None)

celu激活层（CELU Activation Operator）

根据 `Continuously Differentiable Exponential Linear Units <https://arxiv.org/abs/1704.07483>`_ 对输入Tensor中每个元素应用以下计算。

.. math::

    celu(x) = max(0, x) + min(0, \alpha * (e^{x/\alpha} − 1))

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float16、float32、float64。
 - alpha (float, 可选) - celu的alpha值，默认值为1.0。
 - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F

    x = paddle.to_tensor([[-1., 6.], [1., 15.6]])
    out = F.celu(x, alpha=0.2)
    # [[-0.19865242,  6.        ],
    #  [ 1.        , 15.60000038]]
