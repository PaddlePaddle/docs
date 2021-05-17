.. _cn_api_nn_ReLU6:

ReLU6
-------------------------------
.. py:class:: paddle.nn.ReLU6(name=None)

ReLU6激活层

.. math::

    ReLU6(x) = min(max(0,x), 6)

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

    x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
    m = paddle.nn.ReLU6()
    out = m(x) # [0, 0.3, 6]
