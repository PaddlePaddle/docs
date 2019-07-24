.. _cn_api_fluid_layers_sign:

sign
-------------------------------

.. py:function:: paddle.fluid.layers.sign(x)

此函数返回x中每个元素的正负号：1代表正，-1代表负，0代表零。

参数：
    - **x** (Variable|numpy.ndarray) – 输入张量。

返回：输出正负号张量，和x有着相同的形状和数据类型。

返回类型：Variable

**代码示例**

..  code-block:: python

    # [1, 0, -1]
    import paddle.fluid as fluid
    data = fluid.layers.sign(np.array([3, 0, -2]))





