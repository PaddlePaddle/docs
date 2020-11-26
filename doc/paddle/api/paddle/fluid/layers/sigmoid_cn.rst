.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid(x, name=None)




sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数：

    - **x** Tensor - 数据类型为float32，float64。激活函数的输入值。
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None

返回：激活函数的输出值

返回类型：Tensor，数据类型为float32的Tensor。

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.uniform(min=-3., max=3., shape=[3])
    y = paddle.nn.functional.sigmoid(x)
    print(x)
    print(y)









