.. _cn_api_fluid_layers_read_file:

read_file
-------------------------------

.. py:function:: paddle.fluid.layers.read_file(reader)

从给定的reader中读取数据

reader是一个变量，它可以是由函数fluid.layers.open_files()生成的原始reader，或者是由函数fluid.layers.double_buffer()生成的装饰变量，等等。

参数：
    - **reader** (Variable)-待处理的reader

返回：从reader中读取的数据元组，元组数据类型为Variable

返回类型: tuple（元组）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data_file = fluid.layers.open_files(
        filenames=['mnist.recordio'],
        shapes=[(-1, 748), (-1, 1)],
        lod_levels=[0, 0],
        dtypes=["float32", "int64"])
    data_file = fluid.layers.double_buffer(
        fluid.layers.batch(data_file, batch_size=64))
    input, label = fluid.layers.read_file(data_file)









