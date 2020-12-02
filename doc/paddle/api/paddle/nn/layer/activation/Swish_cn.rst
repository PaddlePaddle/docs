.. _cn_api_nn_Swish:

Swish
-------------------------------
.. py:class:: paddle.nn.Swish(name=None)

Swish激活层

.. math::

    Swish(x) = \frac{x}{1 + e^{-x}}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
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

    x = paddle.to_tensor(np.array([-2., 0., 1.]))
    m = paddle.nn.Swish()
    out = m(x) # [-0.238406, 0., 0.731059]
