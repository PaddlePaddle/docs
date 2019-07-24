.. _cn_api_fluid_layers_read_file:

read_file
-------------------------------

.. py:function:: paddle.fluid.layers.read_file(reader)

执行给定的reader变量并从中获取数据

reader也是变量。可以为由fluid.layers.open_files()生成的原始reader或者由fluid.layers.double_buffer()生成的装饰变量，等等。

参数：
    - **reader** (Variable)-将要执行的reader

返回：从给定的reader中读取数据

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









