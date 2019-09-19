.. _cn_api_fluid_layers_sequence_first_step:

sequence_first_step
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_first_step(input)

该层对序列化输入的LoDTensor，选取其每个序列（sequences）的第一个step的特征向量作为池化后的输出向量。

::

    x是1-level LoDTensor:

      x.lod = [[2, 3, 2]]

      x.data = [[1], [3], [2], [4], [6], [5], [1]]

      x.dims = [7, 1]

    输出为张量:

      out.dim = [3, 1]
      with condition len(x.lod[-1]) == out.dims[0]
      out.data = [[1], [2], [5]], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：**input** (variable)- 类型为LoDTensor的输入序列，仅支持lod_level不超过2的LoDTensor。

返回：每个输入序列中的第一个step的特征向量组成的张量

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1], append_batch_size=False,
                 dtype='float32', lod_level=1)
    x_first_step = fluid.layers.sequence_first_step(input=x)









