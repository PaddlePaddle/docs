.. _cn_api_fluid_layers_ceil:

ceil
-------------------------------

.. py:function:: paddle.fluid.layers.ceil(x, name=None)

向上取整运算激活函数。

.. math::
    out = \left \lceil x \right \rceil



参数:

- **x** (Variable) - 该OP的输入为Tensor。数据类型为float32或float64。
- **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回： Ceil算子的输出。

返回类型： Variable - 该OP的输出为Tensor，数据类型为float32，float64。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.ceil(data)
