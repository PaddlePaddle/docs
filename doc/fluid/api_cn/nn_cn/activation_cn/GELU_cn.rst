.. _cn_api_nn_GELU:

GELU
-------------------------------
.. py:class:: paddle.nn.GELU(approximate=False, name=None)

GELU激活层（GELU Activation Operator）

逐元素计算 GELU激活函数。更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ 。

如果使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))


其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - approximate (bool, 可选) - 是否使用近似计算，默认值为 False，即不使用近似计算。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))
    
    m = paddle.nn.GELU()
    out = m(x) # [-0.158655 0.345731 0.841345 1.39979]

    m = paddle.nn.GELU(True)
    out = m(x) # [-0.158808 0.345714 0.841192 1.39957]
