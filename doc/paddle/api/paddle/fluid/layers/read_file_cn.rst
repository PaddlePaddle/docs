.. _cn_api_fluid_layers_read_file:

read_file
-------------------------------


.. py:function:: paddle.fluid.layers.read_file(reader)




从给定的reader中读取数据

reader是一个Variable，它可以是由函数fluid.layers.py_reader()生成的reader，或者是由函数fluid.layers.double_buffer()生成的装饰Variable。

参数：
    - **reader** (Variable)-待处理的reader

返回：从reader中读取的数据元组，元组数据类型为Variable

返回类型: tuple（元组）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    reader = fluid.layers.py_reader(capacity=64,
                                    shapes=[(-1, 1, 28, 28), (-1, 1)],
                                    dtypes=['float32', 'int64'])
    image, label = fluid.layers.read_file(reader)
    data_file = fluid.layers.double_buffer(
        fluid.layers.batch(data_file, batch_size=64))
    input, label = fluid.layers.read_file(data_file)









