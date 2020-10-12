.. _cn_api_nn_Tanhshrink:

Tanhshrink
-------------------------------
.. py:class:: paddle.nn.Tanhshrink(name=None)

Tanhshrink激活层

.. math::

    Tanhshrink(x) = x - tanh(x)

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
    m = paddle.nn.Tanhshrink()
    out = m(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]
