.. _cn_api_fluid_layers_load:

load
-------------------------------

.. py:function:: paddle.fluid.layers.load(out, file_path, load_as_fp16=None)

该OP操作将从磁盘文件中加载LoDTensor/SelectedRows变量。


参数：
    - **out** (Variable) - 需要加载的LoDTensor或SelectedRows。
    - **file_path** (str) - 从“file_path”中加载的变量Variable
    - **load_as_fp16** (BOOLEAN) - 如果为真，张量首先进行加载然后类型转换成float16。如果为假，张量将直接加载，不需要进行数据类型转换。默认为false。

返回：None

**代码示例：**


.. code-block:: python

    import paddle.fluid as fluid
    tmp_tensor = fluid.layers.create_tensor(dtype='float32')
    fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")





