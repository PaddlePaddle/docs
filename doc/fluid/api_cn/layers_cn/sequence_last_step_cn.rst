.. _cn_api_fluid_layers_sequence_last_step:

sequence_last_step
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_last_step(input)

该API可以获取序列的最后一步

::

    x是level-1的LoDTensor:

        x.lod = [[2, 3, 2]]

        x.data = [1, 3, 2, 4, 6, 5, 1]

        x.dims = [7, 1]

    输出为Tensor:

        out.dim = [3, 1]

        且 len(x.lod[-1]) == out.dims[0]

        out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)

参数：**input** (variable)-输入变量，为LoDTensor

返回：序列的最后一步，为张量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_last_step = fluid.layers.sequence_last_step(input=x)









