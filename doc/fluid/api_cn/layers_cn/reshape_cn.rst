.. _cn_api_fluid_layers_reshape:

reshape
-------------------------------

.. py:function::  paddle.fluid.layers.reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None)

保持输入张量数据不变的情况下，改变张量的形状。

目标形状可由 ``shape`` 或 ``actual_shape`` 给出。``shape`` 是一个整数列表，而 ``actual_shape`` 是一个张量变量。
当两个属性同时被指定时，``actual_shape`` 的优先级高于 ``shape`` ，但在编译时仍然应该正确地设置 ``shape`` 以保证形状推断。

在指定目标shape时存在一些技巧：

.. code-block:: text

  1. -1表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
  2. 0表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过秩(x)。


这里有一些例子来解释它们：

.. code-block:: text

  1. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]， ``reshape`` 将x变换为形状为[6,8]的二维张量，且x的数据保持不变。
  2. 给定一个形状为[2,4,6]的三维张量x，指定的目标形状为[2,3,-1,2]， ``reshape``将x变换为形状为[2,3,4,2]的4- d张量，不改变x的数据。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
  3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[- 1,0,3,2]，整形算子将x变换为形状为[2,4,3,2]的四维张量，使x的数据保持不变。在这种情况下，0意味着实际的维值将从x的对应维数中复制,-1位置的维度由x的元素总数和剩余维度计算得来。

参数：
  - **x** (variable) - 输入张量
  - **shape** (list) - 新的形状。新形状最多只能有一个维度为-1。
  - **actual_shape** (variable) - 一个可选的输入。如果提供，则根据 ``actual_shape`` 进行 reshape，而不是指定 ``shape`` 。也就是说，actual_shape具有比shape更高的优先级。
  - **act** (str) - 对reshpe后的tensor变量执行非线性激活
  - **inplace** (bool) - 如果 ``inplace`` 为True，则 ``layers.reshape`` 的输入和输出是同一个变量，否则， ``layers.reshape`` 的输入和输出是不同的变量。请注意，如果x作为多个层的输入，则 ``inplace`` 必须为False。
  - **name** (str) -  可选变量，此层的名称

返回：如果 ``act`` 为 ``None``,返回reshape后的tensor变量。如果 ``inplace`` 为 ``False`` ,将返回一个新的Tensor变量，否则，将改变x自身。如果 ``act`` 不是 ``None`` ，则返回激活的张量变量。

抛出异常：``TypeError`` - 如果 actual_shape 既不是变量也不是None

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(
      name='data', shape=[2, 4, 6], dtype='float32')
  reshaped = fluid.layers.reshape(
      x=data, shape=[-1, 0, 3, 2], inplace=True)










