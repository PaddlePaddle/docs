.. _cn_api_fluid_layers_reshape:

reshape
-------------------------------

.. py:function::  paddle.fluid.layers.reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None)

保持输入张量数据不变的情况下，改变张量的形状。

目标形状可由 ``shape`` 或 ``actual_shape`` 给出。``shape`` 可以是一个包含整数或张量的列表，或者是一个张量变量，而 ``actual_shape`` 是一个张量变量。
当两个属性同时被指定时，``actual_shape`` 的优先级高于 ``shape`` ，但此时 ``shape`` 只能是整数列表，且在编译时仍然应该正确地设置 ``shape`` 以保证形状推断。

在指定目标shape时存在一些技巧：

.. code-block:: text

  1. -1表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
  2. 0表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过秩(x)。


这里有一些例子来解释它们：

.. code-block:: text

  1. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]， ``reshape`` 将x变换为形状为[6,8]的二维张量，且x的数据保持不变。
  2. 给定一个形状为[2,4,6]的三维张量x，指定的目标形状为[2,3,-1,2]， ``reshape``将x变换为形状为[2,3,4,2]的4- d张量，不改变x的数据。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
  3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[- 1,0,3,2]，整形算子将x变换为形状为[2,4,3,2]的四维张量，使x的数据保持不变。在这种情况下，0意味着实际的维值将从x的对应维数中复制,-1位置的维度由x的元素总数和剩余维度计算得来。

**注意:** 参数``actual_shape`` 之后将被舍弃，只用参数 ``shape`` 来表示目标形状。

参数：
  - **x** (Variable) - 输入张量。
  - **shape** (list|tuple|Variable) - 新的形状。新形状最多只能有一个维度为-1。如果 ``shape``是一个 list 或 tuple, 它可以包含整数或者 Variable 类型的元素，但是 Variable 类型元素的形状只能是[1]。
  - **actual_shape** (Variable) - 一个可选的输入。如果提供，则根据 ``actual_shape`` 进行 reshape，而不是指定 ``shape`` 。也就是说，``actual_shape`` 具有比 ``shape`` 更高的优先级，此时 ``shape`` 只能是整数列表。 ``actual_shape`` 将在未来的版本中舍弃。更新提示：``actual_shape`` 将被舍弃并用 ``shape`` 代替。
  - **act** (str) - 对reshpe后的tensor变量执行非线性激活。
  - **inplace** (bool) - 如果 ``inplace`` 为True，则 ``layers.reshape`` 的输入和输出是同一个变量，否则， ``layers.reshape`` 的输入和输出是不同的变量。请注意，如果x作为多个层的输入，则 ``inplace`` 必须为False。
  - **name** (str) -  可选变量，此层的名称。

返回：如果 ``act`` 为 ``None``,返回reshape后的tensor变量。如果 ``inplace`` 为 ``False`` ,将返回一个新的Tensor变量，否则，将改变x自身。如果 ``act`` 不是 ``None`` ，则返回激活的张量变量。

抛出异常：``TypeError`` - 如果 actual_shape 既不是变量也不是None.

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid

  # example 1:
  # attr shape is a list which doesn't contain tensor Variable.
  data_1 = fluid.layers.data(
      name='data_1', shape=[2, 4, 6], dtype='float32')
  reshaped_1 = fluid.layers.reshape(
      x=data_1, shape=[-1, 0, 3, 2], inplace=True)
  # the shape of reshaped_1 is [2,4,3,2].

  # example 2:
  # attr shape is a list which contains tensor Variable.
  data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
  dim = fluid.layers.fill_constant([1], "int32", 5)
  reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])
  # the shape of reshaped_2 is [5,10].










