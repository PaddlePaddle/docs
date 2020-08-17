.. _cn_api_nn_functional_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid(x, name=None)

:alias_main: paddle.nn.functional.sigmoid
:alias: paddle.nn.functional.sigmoid
:update_api: paddle.fluid.layers.sigmoid



sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数：

    - **x** (Tensor)- 数据类型为float32，float64。激活函数的输入值。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：激活函数的输出值

返回类型：Tensor，数据类型为float32的Tensor。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn.functional as F
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_variable(x_data)
    out = F.sigmoid(x)
    print(out.numpy())
    # [0.40131234 0.450166   0.52497919 0.57444252]
