.. _cn_api_nn_ELU:

ELU
-------------------------------
.. py:class:: paddle.nn.ELU(alpha=1.0, name=None)

ELU激活层（ELU Activation Operator）

根据 `Exponential Linear Units <https://arxiv.org/abs/1511.07289>`_ 对输入Tensor中每个元素应用以下计算。

.. math::

    ELU(x) = max(0, x) + min(0, \alpha * (e^{x} − 1))

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - alpha (float, 可选) - ELU的alpha值，默认值为1.0。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
    m = paddle.nn.ELU(0.2)
    out = m(x)
    # [[-0.12642411  6.        ]
    #  [ 1.          15.6      ]]
