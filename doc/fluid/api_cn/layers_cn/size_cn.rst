.. _cn_api_fluid_layers_size:

size
-------------------------------

.. py:function:: paddle.fluid.layers.size(input)

size层

返回一个形为[1]的int64类型的张量，代表着输入张量的元素的数量。


参数：
    - **input** - 输入的变量


返回：输入变量的元素的数量。

**代码示例**：

.. code-block:: python

        import paddle.fluid.layers as layers

        input = layers.data(
                name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
        rank = layers.size(input) # 300