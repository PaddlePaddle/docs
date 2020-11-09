.. _cn_api_nn_Tanh:

Tanh
-------------------------------
.. py:class:: paddle.nn.Tanh(name=None)

Tanh激活层

.. math::
    Tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}


参数
::::::::::
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
    m = paddle.nn.Tanh()
    out = m(x)
    print(out)
    # [-0.37994896 -0.19737532  0.09966799  0.29131261]