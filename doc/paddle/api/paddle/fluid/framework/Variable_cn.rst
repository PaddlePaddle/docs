.. _cn_api_fluid_Variable:

Variable
-------------------------------

.. py:class:: paddle.fluid.Variable




**注意：**
  **1. 请不要直接调用** `Variable` **的构造函数，因为这会造成严重的错误发生！**

  **2. 在静态图形模式下：请使用** `Block.create_var` **创建一个静态的** `Variable` **，该静态的** `Variable` **在使用** :ref:`cn_api_fluid_executor` **执行前是没有实际数据的。**

  **3. 在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下：请使用** :ref:`cn_api_fluid_dygraph_to_variable` 创建一个拥有实际数据的 :ref:`api_guide_Variable`

在Fluid中，OP的每个输入和输出都是 :ref:`api_guide_Variable` 。多数情况下， :ref:`api_guide_Variable` 用于保存不同种类的数据或训练标签。

:ref:`api_guide_Variable` 总是属于某一个 :ref:`api_guide_Block` 。所有 :ref:`api_guide_Variable` 都有其自己的 ``name`` ,不同 :ref:`api_guide_Block` 中的两个 :ref:`api_guide_Variable` 可以具有相同的名称。如果使用的 **不是** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式，那么同一个 :ref:`api_guide_Block` 中的两个或更多 :ref:`api_guide_Variable` 拥有相同 ``name`` 将意味着他们会共享相同的内容。通常我们使用这种方式来实现 **参数共享**

:ref:`api_guide_Variable` 有很多种。它们每种都有自己的属性和用法。请参考 `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_ 以获得详细信息。 :ref:`api_guide_Variable` 的大多数成员变量可以设置为 ``None`` 。它的意思是它不可用或稍后指定。

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
        import numpy as np
        with fluid.dygraph.guard():
            new_variable = fluid.dygraph.to_variable(np.arange(10))

.. py:method:: test()

just for test

.. py:method:: detach()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2.** ``detach`` **后的**  :ref:`api_guide_Variable` **将会成为临时变量**

产生一个新的，和当前计算图分离的，但是拥有当前 :ref:`api_guide_Variable` 其内容的临时变量

返回：一个新的，和当前计算图分离的，但是拥有当前 :ref:`api_guide_Variable` 其内容的临时 :ref:`api_guide_Variable`

返回类型：（:ref:`api_guide_Variable` | 和输入的 ``Dtype`` 一致）

**示例代码**
  .. code-block:: python

     import paddle.fluid as fluid
     from paddle.fluid.dygraph.base import to_variable
     from paddle.fluid.dygraph import Linear
     import numpy as np

     data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
     with fluid.dygraph.guard():
           linear = Linear(32, 64)
           data = to_variable(data)
           x = linear(data)
           y = x.detach()

.. py:method:: numpy()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


返回一个 ``ndarray`` 来表示当前  :ref:`api_guide_Variable` 的值

返回：``numpy`` 的数组，表示当前 :ref:`api_guide_Variable` 的实际值

返回类型：ndarray，``dtype`` 和输入的 ``dtype`` 一致

**示例代码**
  .. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    from paddle.fluid.dygraph import Linear
    import numpy as np

    data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
    with fluid.dygraph.guard():
        linear = Linear(32, 64)
        data = to_variable(data)
        x = linear(data)
        print(x.numpy())

.. py:method:: set_value()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

为此 :ref:`api_guide_Variable` 设置一个新的值。

**参数:**

  - **value**: ( :ref:`api_guide_Variable` 或 ``ndarray`` ) 要赋值给此 :ref:`api_guide_Variable` 的新的值。

返回：无

抛出异常： ``ValueError`` - 当要赋于的新值的 ``shape`` 和此 :ref:`api_guide_Variable` 原有的 ``shape`` 不同时，抛出 ``ValueError`` 。

**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        from paddle.fluid.dygraph.base import to_variable
        from paddle.fluid.dygraph import Linear
        import numpy as np

        data = np.ones([3, 1024], dtype='float32')
        with fluid.dygraph.guard():
            linear = fluid.dygraph.Linear(1024, 4)
            t = to_variable(data)
            linear(t)  # 使用默认参数值调用前向
            custom_weight = np.random.randn(1024, 4).astype("float32")
            linear.weight.set_value(custom_weight)  # 将参数修改为自定义的值
            out = linear(t)  # 使用新的参数值调用前向

.. py:method:: backward()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2. 由于如果该**  :ref:`api_guide_Variable` **以上没有任何地方需要梯度，那么仅仅设置该**  :ref:`api_guide_Variable` **的梯度为** ``1`` **是没有意义的。因此，这种情况下，为了节省一些计算，我们不去产生该** :ref:`api_guide_Variable` **的梯度**

从该节点开始执行反向

**参数:**

  - **retain_graph** (bool，可选) – 该参数用于确定反向梯度更新完成后反向梯度计算图是否需要保留（retain_graph为True则保留反向梯度计算图）。若用户打算在执行完该方法（  :code:`backward` ）后，继续向之前已构建的计算图中添加更多的Op，则需要设置 :code:`retain_graph` 值为True（这样才会保留之前计算得到的梯度）。可以看出，将 :code:`retain_graph` 设置为False可降低内存的占用。默认值为False。

返回：无


**示例代码**
  .. code-block:: python

        import numpy as np
        import paddle
        paddle.disable_static()
        x = np.ones([2, 2], np.float32)
        inputs = []
        for _ in range(10):
            tmp = paddle.to_tensor(x)
            # 如果这里我们不为输入tmp设置stop_gradient=False，那么后面loss也将因为这个链路都不需要梯度
            # 而不产生梯度
            tmp.stop_gradient=False
            inputs.append(tmp)
        ret = paddle.sums(inputs)
        loss = paddle.reduce_sum(ret)
        loss.backward()

.. py:method:: gradient()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

  **2. 由于如果该**  :ref:`api_guide_Variable` **以上没有任何地方需要梯度，那么仅仅设置该**  :ref:`api_guide_Variable` **的梯度为** ``1`` **是没有意义的。因此，这种情况下，为了节省一些计算，我们不去产生该** :ref:`api_guide_Variable` **的梯度**

获取该 :ref:`api_guide_Variable` 的梯度值

返回：如果 :ref:`api_guide_Variable` 的类型是LoDTensor（参见 :ref:`cn_user_guide_lod_tensor` ），返回该 :ref:`api_guide_Variable` 类型为 ``ndarray`` 的梯度值；如果 :ref:`api_guide_Variable` 的类型是SelectedRows，返回该 :ref:`api_guide_Variable` 类型为 ``ndarray`` 的梯度值和类型为 ``ndarray`` 的词id组成的tuple。

返回类型：``ndarray`` 或者 ``tuple of ndarray`` , 返回类型 ``tuple of ndarray`` 仅在 :ref:`cn_api_fluid_dygraph_Embedding` 层稀疏更新时产生。


**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        # example1: 返回ndarray
        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            inputs2 = []
            for _ in range(10):
                tmp = fluid.dygraph.base.to_variable(x)
                tmp.stop_gradient=False
                inputs2.append(tmp)
            ret2 = fluid.layers.sums(inputs2)
            loss2 = fluid.layers.reduce_sum(ret2)
            loss2.backward()
            print(loss2.gradient())

        # example2: 返回tuple of ndarray
        with fluid.dygraph.guard():
            embedding = fluid.dygraph.Embedding(
                size=[20, 32],
                param_attr='emb.w',
                is_sparse=True)
            x_data = np.arange(12).reshape(4, 3).astype('int64')
            x_data = x_data.reshape((-1, 3, 1))
            x = fluid.dygraph.base.to_variable(x_data)
            out = embedding(x)
            out.backward()
            print(embedding.weight.gradient())

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
            loss2.backward()
            print(loss2.gradient())
            loss2.clear_gradient()
            print("After clear {}".format(loss2.gradient()))


.. py:method:: to_string()

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

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


.. py:method:: astype(self, dtype)

将该 :ref:`api_guide_Variable` 中的数据转换成目标 ``Dtype``

**参数：**
 - **self** ( :ref:`api_guide_Variable` ) - 当前 :ref:`api_guide_Variable` ， 用户不需要传入。
 - **dtype** (int | float | float64) - 希望转换成的 ``Dtype``


返回：一个全新的转换了 ``Dtype`` 的 :ref:`api_guide_Variable`

返回类型： :ref:`api_guide_Variable`


**示例代码**

在静态图模式下：
    .. code-block:: python

        import paddle.fluid as fluid

        startup_prog = fluid.Program()
        main_prog = fluid.Program()
        with fluid.program_guard(startup_prog, main_prog):
            original_variable = fluid.data(name = "new_variable", shape=[2,2], dtype='float32')
            new_variable = original_variable.astype('int64')
            print("new var's dtype is: {}".format(new_variable.dtype))


在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下：
    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        x = np.ones([2, 2], np.float32)
        with fluid.dygraph.guard():
            original_variable = fluid.dygraph.to_variable(x)
            print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
            new_variable = original_variable.astype('int64')
            print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))



属性
::::::::::::

.. py:attribute:: stop_gradient

**注意：该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下除参数以外默认值为** ``True`` **，而参数的该属性默认值为** ``False`` **。在静态图下所有的** :ref:`api_guide_Variable` **该属性默认值都为** ``False``

是否从此 :ref:`api_guide_Variable` 开始，之前的相关部分都停止梯度计算

**示例代码**
  .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = fluid.Linear(13, 5, dtype="float32")
            linear2 = fluid.Linear(3, 3, dtype="float32")
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            out.backward()
            # 可以发现这里linear的参数梯度变成了None
            assert linear.weight.gradient() is None
            assert out1.gradient() is None

.. py:attribute:: persistable

**注意：该属性我们即将废弃，此介绍仅为了帮助用户理解概念， 1.6版本后用户可以不再关心该属性**

  **1. 该属性除参数以外默认值为** ``False`` **，而参数的该属性默认值为** ``True`` 。

  **2. 该属性在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下一经初始化即不能修改，这是由于在动态执行时，**  :ref:`api_guide_Variable` **的生命周期将由** ``Python`` **自行控制不再需要通过该属性来修改**

此 :ref:`api_guide_Variable` 是否是长期存活的 :ref:`api_guide_Variable`

.. py:attribute:: name

**注意：在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下，那么同一个** :ref:`api_guide_Block` **中的两个或更多** :ref:`api_guide_Variable` **拥有相同** ``name`` **将意味着他们会共享相同的内容。通常我们使用这种方式来实现参数共享**

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

此 :ref:`api_guide_Variable` 的 ``LoD`` 信息，关于 ``LoD`` 可以参考 :ref:`api_fluid_LoDTensor` 相关内容

.. py:attribute:: type

**注意：该属性是只读属性**

此 :ref:`api_guide_Variable` 的内存模型，例如是：:ref:`api_fluid_LoDTensor`， 或者SelectedRows
