.. _cn_api_nn_LeakyReLU:

LeakyReLU
-------------------------------
.. py:class:: paddle.nn.LeakyReLU(negative_slope=0.01, name=None)

LeakyReLU 激活层

.. math::

    LeakyReLU(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x >= 0 \\
        &negative\_slope * x, & & otherwise \\
        \end{aligned}
        \right. \\

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - negative_slope (float，可选) - :math:`x < 0` 时的斜率。默认值为0.01。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    m = paddle.nn.LeakyReLU()
    x = paddle.to_tensor(np.array([-2, 0, 1], 'float32'))
    out = m(x)  # [-0.02, 0., 1.]
