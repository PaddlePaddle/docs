.. _cn_api_tensor_sin:

sin
-------------------------------

.. py:function:: paddle.sin(x, name=None)

:alias_main: paddle.sin
:alias: paddle.sin,paddle.tensor.sin,paddle.tensor.math.sin
:update_api: paddle.fluid.layers.sin



计算输入的正弦值。

.. math::
        out = sin(x)

参数:
    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：返回类型为Tensor， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_variable(x_data)
    out = paddle.sin(x)
    print(out.numpy())
    # [-0.38941834 -0.19866933  0.09983342  0.29552021]
