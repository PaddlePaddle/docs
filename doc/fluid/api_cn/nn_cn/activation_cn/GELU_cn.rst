.. _cn_api_fluid_layers_gelu:

GELU
-------------------------------
.. py:class:: paddle.nn.GELU(approximate=False, name=None)

GELU激活层（GELU Activation Operator）

更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`。

如果使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    GELU(x) = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))


其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - approximate (bool, 可选) - 是否使用近似计算，默认值为 False。
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
    data = np.random.randn(2, 3).astype("float32")
    x = paddle.to_tensor(data)

    m = paddle.nn.GELU()
    out = m(x) 
    
    data
    # array([[ 0.87165993, -1.0541513 , -0.37214822],
    #         [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
    out
    # array([[ 0.70456535, -0.15380788, -0.13207214],
    #        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)
