.. _cn_api_nn_Softsign:

Softsign
-------------------------------
.. py:class:: paddle.nn.Softsign(name=None)

Softsign激活层

.. math::

    Softsign(x) = \frac{x}{1 + |x|}

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

    x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
    m = paddle.nn.Softsign()
    out = m(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]
