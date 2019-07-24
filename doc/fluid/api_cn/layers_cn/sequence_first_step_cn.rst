.. _cn_api_fluid_layers_sequence_first_step:

sequence_first_step
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_first_step(input)

该功能获取序列的第一步

::

    x是1-level LoDTensor:

      x.lod = [[2, 3, 2]]

      x.data = [1, 3, 2, 4, 6, 5, 1]

      x.dims = [7, 1]

    输出为张量:

      out.dim = [3, 1]
      with condition len(x.lod[-1]) == out.dims[0]
      out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列第一步，为张量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_first_step = fluid.layers.sequence_first_step(input=x)









