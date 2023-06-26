.. _cn_api_fluid_Variable:

Variable
-------------------------------

.. py:class:: paddle.static.Variable(block, type=<VarType.LOD_TENSOR: 7>, name=None, shape=None, dtype=None, lod_level=None, capacity=None, persistable=None, error_clip=None, stop_gradient=False, is_data=False, need_check_feed=False, belong_to_optimizer=False, **kwargs)


.. note::
  1. 请不要直接调用 `Variable` 的构造函数，因为这会造成严重的错误发生！

  2. 在静态图形模式下：请使用 `Block.create_var` 创建一个静态的 `Variable`，该静态的 `Variable` 在使用 :ref:`cn_api_fluid_executor` 执行前是没有实际数据的。

在 Paddle 静态图模式中，OP 的每个输入和输出都是 :ref:`api_guide_Variable`。多数情况下，:ref:`api_guide_Variable` 用于保存不同种类的数据或训练标签。

:ref:`api_guide_Variable` 总是属于某一个 :ref:`api_guide_Block`。所有 :ref:`api_guide_Variable` 都有其自己的 ``name``，不同 :ref:`api_guide_Block` 中的两个 :ref:`api_guide_Variable` 可以具有相同的名称。如果使用的 **不是** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式，那么同一个 :ref:`api_guide_Block` 中的两个或更多 :ref:`api_guide_Variable` 拥有相同 ``name`` 将意味着他们会共享相同的内容。通常我们使用这种方式来实现 **参数共享**。

:ref:`api_guide_Variable` 有很多种。它们每种都有自己的属性和用法。请参考 `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_ 以获得详细信息。:ref:`api_guide_Variable` 的大多数成员变量可以设置为 ``None``。它的意思是它不可用或稍后指定。

如果您希望创建一个 :ref:`api_guide_Variable` 那么可以参考如下示例：

代码示例
::::::::::::
COPY-FROM: paddle.static.Variable:code-example1

方法
::::::::::::
to_string(throw_on_error, with_details=True)
'''''''''

获取该 :ref:`api_guide_Variable` 的静态描述字符串。

**参数：**

 - **throw_on_error** (bool) - 是否在没有设置必需字段时抛出异常。
 - **with_details** (bool) - 值为 true 时，打印更多关于 :ref:`api_guide_Variable` 的信息，如 ``error_clip`` , ``stop_gradient`` 等。

**返回**

用于静态描述该 :ref:`api_guide_Variable` 的字符串。


**代码示例**
COPY-FROM: paddle.static.Variable.to_string


clone(self)
'''''''''

返回一个新的 ``Variable``，其复制原 ``Variable`` 并且新的 ``Variable`` 也被保留在计算图中，即复制的新 ``Variable`` 也参与反向计算。调用 ``out = variable.clone()`` 与 ``out = assign(variable)`` 效果一样。

**返回**

复制的新 ``Variable``。

**代码示例**
COPY-FROM: paddle.static.Variable.clone


detach(self)
'''''''''

返回一个新的 ``Variable``，并从当前计算图分离。

**返回**

与当前计算图分离的 ``Variable``。

**代码示例**
COPY-FROM: paddle.static.Variable.detach


astype(self, dtype)
'''''''''

将该 :ref:`api_guide_Variable` 中的数据转换成目标 ``Dtype``。

**参数：**

 - **self** ( :ref:`api_guide_Variable` ) - 当前 :ref:`api_guide_Variable`，用户不需要传入。
 - **dtype** (int | float | float64) - 希望转换成的 ``Dtype``。


**返回**
一个全新的转换了 ``Dtype`` 的 :ref:`api_guide_Variable`。



**代码示例**
COPY-FROM: paddle.static.Variable.astype


get_value(scope=None)
'''''''''

获取 :ref:`api_guide_Variable` 的值。

**参数**

  - scope ( Scope，可选 ) - 从指定的 ``scope`` 中获取 :ref:`api_guide_Variable` 的值。如果 ``scope`` 为 ``None``，通过 `paddle.static.global_scope()` 获取全局/默认作用域实例，并从中获取 :ref:`api_guide_Variable` 的值；否则，从指定的 ``scope`` 中获取 :ref:`api_guide_Variable` 的值。

**返回**

Tensor， :ref:`api_guide_Variable` 的值。

**代码示例**

**代码示例**
COPY-FROM: paddle.static.Variable.get_value


set_value(value, scope=None)
'''''''''

将 ``value`` 设置为 :ref:`api_guide_Variable` 的值。

**参数**

  - value ( Tensor|ndarray ) - :ref:`api_guide_Variable` 的值。
  - scope ( Scope，可选 ) - 将 :ref:`api_guide_Variable` 的值设置到指定的 ``scope`` 中。如果 ``scope`` 为 ``None``，通过 `paddle.static.global_scope()` 获取全局/默认作用域实例，并将 :ref:`api_guide_Variable` 的值设置到这个用域实例中；否则，将 :ref:`api_guide_Variable` 的值设置到指定的 ``scope`` 中。

**返回**

无。

**代码示例**

COPY-FROM: paddle.static.Variable.set_value


size(self)
'''''''''

返回该 :ref:`api_guide_Variable` 中的数据元素数量，结果是一个 shape 为[1]的 int64 的 ``Variable`` 。

**返回**

``Variable``：单元元素数量。


**代码示例**
COPY-FROM: paddle.static.Variable.size


ndimension(self)
'''''''''

返回该 :ref:`api_guide_Variable` 的维度，也称作 rank。

**返回**

``Variable`` 的维度。

**代码示例**
  .. code-block:: python

      import paddle

      paddle.enable_static()

      x = paddle.static.data(name="x", shape=[10, 10], dtype='float32')
      print("Variable's number of dimension: ", x.ndimension())
      # Variable's number of dimension: 2


dim(self)
'''''''''

返回该 :ref:`api_guide_Variable` 的维度，也称作 rank。

**返回**
``Variable`` 的维度。

**代码示例**
  .. code-block:: python

      import paddle

      paddle.enable_static()

      x = paddle.static.data(name="x", shape=[10, 10], dtype='float32')
      print("Variable's number of dim: ", x.dim())
      # Variable's number of dim: 2


属性
::::::::::::
persistable
'''''''''

.. note::
1. 该属性我们即将废弃，此介绍仅为了帮助用户理解概念，1.6 版本后用户可以不再关心该属性
2. 该属性除参数以外默认值为 ``False``，而参数的该属性默认值为 ``True`` 。

此 :ref:`api_guide_Variable` 是否是长期存活的 :ref:`api_guide_Variable`。

name
'''''''''

.. note::
静态图模式下，同一个 :ref:`api_guide_Block` 中的两个或更多 :ref:`api_guide_Variable` 拥有相同 ``name`` 将意味着他们会共享相同的内容。通常我们使用这种方式来实现参数共享。

此 :ref:`api_guide_Variable` 的名字（str）。


shape
'''''''''

.. note::
该属性是只读属性。

此 :ref:`api_guide_Variable` 在每个维度上的元素数量。

dtype
'''''''''

.. note::
该属性是只读属性。

此 :ref:`api_guide_Variable` 的实际数据类型。

lod_level
'''''''''

.. note::
该属性是只读属性。

此 :ref:`api_guide_Variable` 的 ``LoD`` 信息，关于 ``LoD`` 可以参考 :ref:`api_fluid_LoDTensor` 相关内容。

type
'''''''''

.. note::
该属性是只读属性。

此 :ref:`api_guide_Variable` 的内存模型，例如是：:ref:`api_fluid_LoDTensor`，或者 SelectedRows。

ndim
'''''''''

.. note::
该属性是只读属性。

此 :ref:`api_guide_Variable` 的维度，也称作 rank。
