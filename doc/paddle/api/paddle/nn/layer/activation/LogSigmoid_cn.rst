.. _cn_api_nn_LogSigmoid:

LogSigmoid
-------------------------------
.. py:class:: paddle.nn.LogSigmoid(name=None)

LogSigmoid激活层。计算公式如下：

.. math::

    LogSigmoid(x) = \log \frac{1}{1 + e^{-x}}

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

    x = paddle.to_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    m = paddle.nn.LogSigmoid()
    out = m(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]
