.. _cn_api_fluid_layers_size:

size
-------------------------------

.. py:function:: paddle.fluid.layers.size(input)

返回张量的单元数量，是一个shape为[1]的int64的张量。

参数:
    - **input** （Variable）- 输入变量

返回：(Variable)。

**代码示例**：

.. code-block:: python

        import paddle.fluid.layers as layers

        input = layers.data(
            name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
        rank = layers.size(input) # 300













