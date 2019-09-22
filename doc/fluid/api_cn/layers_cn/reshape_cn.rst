.. _cn_api_fluid_layers_reshape:

reshape
-------------------------------

.. py:function::  paddle.fluid.layers.reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None)

该OP在保持输入 ``x`` 数据不变的情况下，改变 ``x`` 的形状。

目标形状可由 ``shape`` 或 ``actual_shape`` 给出。当两个属性同时被指定时，``actual_shape`` 的优先级高于 ``shape`` ，但此时 ``shape`` 只能是整数列表或元组，且在编译时仍然应该正确地设置 ``shape`` 以保证形状推断。

在指定目标shape时存在一些技巧：

.. code-block:: text

  1. -1 表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
  2. 0 表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过x的维度。


这里有一些例子来解释它们：

.. code-block:: text

  1. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]，则将x变换为形状为[6,8]的2-D张量，且x的数据保持不变。
  2. 给定一个形状为[2,4,6]的三维张量x，目标形状为[2,3,-1,2]，则将x变换为形状为[2,3,4,2]的4-D张量，且x的数据保持不变。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
  3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[-1,0,3,2]，则将x变换为形状为[2,4,3,2]的4-D张量，且x的数据保持不变。在这种情况下，0对应位置的维度值将从x的对应维数中复制,-1对应位置的维度值由x的元素总数和剩余维度推断出来。

**注意：参数** ``actual_shape`` **之后将被舍弃，只用参数** ``shape`` **来表示目标形状。**

参数：
  - **x** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32``，或 ``int64``。
  - **shape** （list|tuple|Variable）- 数据类型是 ``int32`` 。定义目标形状。目标形状最多只能有一个维度为-1。如果 ``shape`` 的类型是 list 或 tuple, 它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``shape`` 的类型是 ``Variable``，则是1-D的 ``Tensor`` 或 ``LoDTensor``。
  - **actual_shape** （Variable，可选）- 1-D ``Tensor`` 或 ``LoDTensor``，默认值：`None`。如果 ``actual_shape`` 被提供，``actual_shape`` 具有比 ``shape`` 更高的优先级，此时 ``shape`` 只能是整数列表或元组。更新提示：``actual_shape`` 在未来的版本中将被舍弃，并用 ``shape`` 代替。
  - **act** （str，可选）- 对形状改变后的输入变量做非线性激活操作，激活函数类型可以参考 :ref:`api_guide_activations` 。默认值： ``None``。
  - **inplace** （bool，可选）- 如果 ``inplace`` 为 ``True``，则 ``layers.reshape`` 的输入和输出是同一个变量，否则 ``layers.reshape`` 的输入和输出是不同的变量。默认值：``False``。请注意，如果 ``x`` 是多个OP的输入，则 ``inplace`` 必须为False。
  - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`。默认值： ``None``。

返回：多维 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``input`` 相同。如果 ``inplace`` 为 ``False``，则返回一个新的变量，否则将改变输入变量 ``x`` 自身。如果 ``act`` 为 ``None``，则直接返回形状改变后的变量，否则返回经过激活函数后的变量。

返回类型：Variable。

抛出异常：
    - :code:`TypeError`：``actual_shape`` 的类型应该是 Variable 或 None。
    - :code:`TypeError`：``starts`` 的类型应该是list、tuple 或 Variable。
    - :code:`ValueError`：``shape`` 中至多有一个元素可以是-1。
    - :code:`ValueError`：``shape`` 中的元素为0时，对应的维度应该小于等于``x``的维度。
    - :code:`ValueError`：``shape`` 中的元素除了-1之外，都应该是非负值。

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










