.. _cn_api_fluid_Variable:

Variable
-------------------------------

.. py:class:: paddle.fluid.Variable

**注意：**
  **1. 请不要直接调用** `Variable` **的构造函数，因为这会造成严重的错误发生！**

  **2. 在静态图形模式下：请使用** `Block.create_var` **创建一个静态的** `Variable` **，该静态的** `Variable` **在使用** :ref:`cn_api_fluid_executor` **执行前是没有实际是数据的。**

  **3. 在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下：请使用** :ref:`cn_api_fluid_dygraph_to_variable` 创建一个拥有实际数据的 :ref:`api_guide_Variable`

在Fluid中，OP的每个输入和输出都是 :ref:`api_guide_Variable` 。多数情况下， :ref:`api_guide_Variable` 用于保存不同种类的数据或训练标签。

:ref:`api_guide_Variable` 总是属于某一个 :ref:`api_guide_Block` 。所有 :ref:`api_guide_Variable` 都有其自己的 ``name`` ,不同 :ref:`api_guide_Block` 中的两个 :ref:`api_guide_Variable` 可以具有相同的名称。如果使用的 **不是** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式，那么同一个 :ref:`api_guide_Block` 中的两个或更多 :ref:`api_guide_Variable` 拥有相同 ``name`` 将意味着他们会共享相同的内容。通常我们使用这种方式来实现 **参数共享**

:ref:`api_guide_Variable` 有很多种。它们每种都有自己的属性和用法。请参考 `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_ 以获得详细信息。 :ref:`api_guide_Variable` 的大多数成员变量可以设置为 ``None``。它的意思是它不可用或稍后指定。

如果您希望创建一个 :ref:`api_guide_Variable` 那么可以参考如下示例：

**示例代码：**

        在静态图形模式下：

        .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
        在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下：

        .. code-block:: python

            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                new_variable = fluid.dygraph.to_variable(np.arange(10))


.. py:method:: detach()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2.** ``detach`` **后的**  :ref:`api_guide_Variable` **将会成为临时变量**

返回一个新的，和当前计算图分离的，但是拥有当前 :ref:`api_guide_Variable` 其内容的临时变量

返回：一个新的，和当前计算图分离的，但是拥有当前 :ref:`api_guide_Variable` 其内容的临时 :ref:`api_guide_Variable`

返回类型：（:ref:`api_guide_Variable` | 和输入的 ``Dtype`` 一致）

**示例代码**
  .. code-block:: python

     import paddle.fluid as fluid
     from paddle.fluid.dygraph.base import to_variable
     from paddle.fluid.dygraph import FC
     import numpy as np

     data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
     with fluid.dygraph.guard():
           fc = FC("fc", 64, num_flatten_dims=2)
           data = to_variable(data)
           x = fc(data)
           y = x.detach()

.. py:method:: numpy()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


返回一个 ``ndarray`` 来表示当前  :ref:`api_guide_Variable` 的值

返回：``numpy`` 的数组，表示当前 :ref:`api_guide_Variable` 的实际值

返回类型：ndarray，``dtype`` 和输入的 ``Dtype`` 一致

**示例代码**
  .. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    from paddle.fluid.dygraph import FC
    import numpy as np

    data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
    with fluid.dygraph.guard():
        fc = FC("fc", 64, num_flatten_dims=2)
        data = to_variable(data)
        x = fc(data)
        print(x.numpy())

.. py:method:: backward()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2. 由于如果该**  :ref:`api_guide_Variable` **以上没有任何地方需要梯度，那么仅仅设置该**  :ref:`api_guide_Variable` **的梯度为** ``1`` **是没有意义的。因此，这种情况下，为了节省一些计算，我们不去产生该** :ref:`api_guide_Variable` **的梯度**

从该节点开始执行反向

**参数:**

  - **backward_strategy**: ( :ref:`cn_api_fluid_dygraph_BackwardStrategy` ) 使用何种 :ref:`cn_api_fluid_dygraph_BackwardStrategy`  聚合反向的梯度

返回：无


**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient=False
                inputs2.append(tmp)
            ret2 = fluid.layers.sums(inputs2)
            loss2 = fluid.layers.reduce_sum(ret2)
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            loss2.backward(backward_strategy)

.. py:method:: gradeint()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2. 由于如果该**  :ref:`api_guide_Variable` **以上没有任何地方需要梯度，那么仅仅设置该**  :ref:`api_guide_Variable` **的梯度为** ``1`` **是没有意义的。因此，这种情况下，为了节省一些计算，我们不去产生该** :ref:`api_guide_Variable` **的梯度**

获取该 :ref:`api_guide_Variable` 的梯度值

返回：该 :ref:`api_guide_Variable` 的梯度 ``ndarray`` 值

返回类型：``ndarray``


**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient=False
                inputs2.append(tmp)
            ret2 = fluid.layers.sums(inputs2)
            loss2 = fluid.layers.reduce_sum(ret2)
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            loss2.backward(backward_strategy)
            print(loss2.gradient())

.. py:method:: clear_gradient()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2. 只有当该** :ref:`api_guide_Variable` **有梯度时才可调用，通常我们都会为参数调用这个方法，因为临时变量的梯度将会在其离开作用域时被** ``python`` **自动清除**

设置该 :ref:`api_guide_Variable` 的梯度为零

返回：无


**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient=False
                inputs2.append(tmp)
            ret2 = fluid.layers.sums(inputs2)
            loss2 = fluid.layers.reduce_sum(ret2)
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            loss2.backward(backward_strategy)
            print(loss2.gradient())
            loss2.clear_gradient()
            print("After clear {}".format(loss2.gradient()))


.. py:method:: to_string()

**注意：**

  **1. 该API参数只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

获取该 :ref:`api_guide_Variable` 的静态描述字符串

**参数：（仅在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效）**
 - **throw_on_error** (bool) - 是否在没有设置必需字段时抛出异常。
 - **with_details** (bool) - 值为true时，打印更多关于 :ref:`api_guide_Variable` 的信息，如 ``error_clip`` , ``stop_gradient`` 等


返回：用于静态描述该 :ref:`api_guide_Variable` 的字符串


返回： 将Program转换为字符串

返回类型： str

抛出异常： ``ValueError`` - 当 ``throw_on_error == true`` ，当没有设置任何必需的字段时，抛出 ``ValueError`` 。


**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid

        cur_program = fluid.Program()
        cur_block = cur_program.current_block()
        new_variable = cur_block.create_var(name="X",
                                            shape=[-1, 23, 48],
                                            dtype='float32')
        print(new_variable.to_string(True))
        print("\n=============with detail===============\n")
        print(new_variable.to_string(True, True))


Properties
::::::::::::

.. py:attribute:: stop_gradient

**注意：该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下默认值为** ``True`` **，而参数的该属性默认值为** ``False`` **。在静态图下默认值为** ``False``

是否从此 :ref:`api_guide_Variable` 开始，之前的相关部分都停止梯度计算

**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            fc = fluid.FC("fc1", size=5, dtype="float32")
            fc2 = fluid.FC("fc2", size=3, dtype="float32")
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = fc(a)
            out2 = fc2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            out.backward()
            # 可以发现这里fc的参数变成了
            assert (fc._w.gradient() == 0).all()
            assert (out1.gradient() == 0).all()

.. py:attribute:: persistable

**注意：**

  **1. 该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下默认值为** ``False`` **，而参数的该属性默认值为** ``True`` **。在静态图下默认值为** ``False``

  **2. 该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下一经初始化即不能修改，这是由于在动态执行时，**  :ref:`api_guide_Variable` **的生命周期将由** ``Python`` **自行控制不再需要通过该属性来修改**

此 :ref:`api_guide_Variable` 是否是长期存活的 :ref:`api_guide_Variable`

.. py:attribute:: name

**注意：**

  **1. 在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下，那么同一个** :ref:`api_guide_Block` **中的两个或更多** :ref:`api_guide_Variable` **拥有相同** ``name`` **将意味着他们会共享相同的内容。通常我们使用这种方式来实现参数共享**

  **2. 该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下一经初始化即不能修改，这是由于在动态执行时，**  :ref:`api_guide_Variable` **的生命周期将由** ``Python`` **自行控制不再需要通过该属性来修改**

此 :ref:`api_guide_Variable` 的名字（str）


.. py:attribute:: shape

**注意：该属性是只读属性**

此 :ref:`api_guide_Variable` 的维度

.. py:attribute:: dtype

**注意：该属性是只读属性**

此 :ref:`api_guide_Variable` 的实际数据类型

.. py:attribute:: lod_level

**注意：**

  **1. 该属性是只读属性**

  **2.** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下，不支持该属性，该值为零**

此 :ref:`api_guide_Variable` 的 ``LOD`` 信息，关于 ``LOD`` 可以参考 :ref:`api_fluid_LoDTensor` 相关内容

.. py:attribute:: type

**注意：该属性是只读属性**

此 :ref:`api_guide_Variable` 的内存模型，例如是：:ref:`api_fluid_LoDTensor`， 或者SelectedRows