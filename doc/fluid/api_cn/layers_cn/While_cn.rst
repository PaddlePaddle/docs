.. _cn_api_fluid_layers_While:

While
-------------------------------

.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)


该类用于实现while循环控制功能。


参数：
    - **cond** (Variable) – 用于比较的条件
    - **is_test** (bool) – 用于表明是不是在测试阶段执行
    - **name** (str) - 该层的命名

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
  d0 = fluid.layers.data("d0", shape=[10], dtype='float32')
  data_array = fluid.layers.array_write(x=d0, i=i)
  array_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=3)

  cond = fluid.layers.less_than(x=i, y=array_len)
  while_op = fluid.layers.While(cond=cond)
  with while_op.block():
      d = fluid.layers.array_read(array=data_array, i=i)
      i = fluid.layers.increment(x=i, value=1, in_place=True)
      
      fluid.layers.less_than(x=i, y=array_len, cond=cond)











