.. _cn_api_fluid_layers_sums:

sums
-------------------------------

.. py:function:: paddle.fluid.layers.sums(input,out=None)

该函数对输入进行求和，并返回求和结果作为输出。

参数：
    - **input** (Variable|list)-输入张量，有需要求和的元素
    - **out** (Variable|None)-输出参数。求和结果。默认：None

返回：输入的求和。和参数'out'等同

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    # sum of several tensors
    a0 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
    a1 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
    a2 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=3)
    sums = fluid.layers.sums(input=[a0, a1, a2])

    # sum of a tensor array
    array = fluid.layers.create_array('int64')
    i = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)
    fluid.layers.array_write(a0, array=array, i=i)
    i = fluid.layers.increment(x=i)
    fluid.layers.array_write(a1, array=array, i=i)
    i = fluid.layers.increment(x=i)
    fluid.layers.array_write(a2, array=array, i=i)
    sums = fluid.layers.sums(input=array)









