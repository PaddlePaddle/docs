.. _cn_api_fluid_Program:

Program
-------------------------------

.. py:class::  paddle.fluid.Program




**注意：默认情况下，Paddle Fluid内部默认含有** :ref:`cn_api_fluid_default_startup_program` **和** :ref:`cn_api_fluid_default_main_program` **，它们共享参数。** :ref:`cn_api_fluid_default_startup_program` **只运行一次来初始化参数，** :ref:`cn_api_fluid_default_main_program` **在每个mini batch中运行并更新权重。**

Program是Paddle Fluid对于计算图的一种静态描述，使用Program的构造函数可以创建一个Program。Program中包括至少一个 :ref:`api_guide_Block` ，当 :ref:`api_guide_Block` 中存在条件选择的控制流OP（例如 :ref:`cn_api_fluid_layers_While` 等）时，该Program将会含有嵌套着的 :ref:`api_guide_Block` 即控制流外部的 :ref:`api_guide_Block` 将包含着控制流内部的 :ref:`api_guide_Block` ，而嵌套的 :ref:`api_guide_Block` 的元素访问控制将由具体的控制流OP来决定。关于Program具体的结构和包含的类型请参阅 `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_
。

一个Program的集合通常包含初始化程序（startup_program）与主程序(main_program)，初始化程序是一个包含一些初始化工作的Program，主程序将会包含用来训练的网络结构和变量，在使用同一个 :ref:`api_guide_executor` 执行时他们会共享初始化工作的结果，例如初始化的参数。一个Program的集合可以被用来测试或者训练，被用来训练时， ``Paddle Fluid`` 将会利用所有用户使用的OP和变量来搭建一个训练网络，被用来测试时， 可以通过调用Program相关的接口例如：`clone` 剪去一些与测试无关的OP和变量，比如反向传播的OP和变量。


返回：创建的空的Program

返回值类型：Program

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: to_string(throw_on_error, with_details=False)

将Program转换为字符串

参数：
 - **throw_on_error** (bool) - 是否在没有设置必需字段时抛出异常。
 - **with_details** (bool) - 值为true时，打印更多关于变量和参数的信息，如trainable, optimize_attr等

返回： 将Program转换为字符串

返回类型： str

抛出异常： ``ValueError`` - 当 ``throw_on_error == true`` ，当没有设置任何必需的字段时，抛出 ``ValueError`` 。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: clone(for_test=False)

**注意:**
    **1.** ``Program.clone()`` **方法不会克隆例如**  :ref:`cn_api_fluid_io_DataLoader` **这样的数据读取相关的部分，这可能会造成的数据读取部分在克隆后丢失**

    **2. 此API当** ``for_test=True`` **时将会裁剪部分OP和变量。为防止错误的裁剪，推荐在** :ref:`cn_api_fluid_backward_append_backward` **和执行优化器之前使用** ``clone(for_test=True)`` 。


当 ``for_test=True`` 时创建一个新的、仅包含当前Program前向内容的Program。否则创建一个新的，和当前Program完全相同的Program

有些OP，在训练和测试之间的行为是不同的，比如  :ref:`cn_api_fluid_layers_batch_norm` 。它们有一个属性 ``is_test`` 来控制行为。当 ``for_test=True`` 时，此方法将把它们的 ``is_test`` 属性更改为True。

- 克隆Program用于训练时，将 ``for_test`` 设置为False。
- 克隆Program用于测试时，将 ``for_test`` 设置为True。虽然在这种情况下，如果在使用了优化器之后调用 ``clone`` 我们依旧会对Program当中反向执行以及优化器相关的内容进行自动裁剪，但是，我们强烈建议在使用优化器之前使用 ``clone`` 例如如果使用的是 :ref:`cn_api_fluid_optimizer_Momentum` 可以这样去使用:

**代码示例**

   ::

        import paddle.fluid as fluid
        img = fluid.layers.data(name='image', shape=[784])
        pred = fluid.layers.fc(input=img, size=10, act='relu')
        loss = fluid.layers.mean(pred)
        ## 我们推荐在使用 Optimizer前使用clone()接口
        test_program = fluid.default_main_program().clone(for_test=True)
        optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
        optimizer.minimize(loss)

参数：
 - **for_test** (bool) – 取值为True时，clone方法内部会把operator的属性 ``is_test`` 设置为 True， 并裁剪反向OP和参数优化OP，默认值为False

返回：当 ``for_test=True`` 时返回一个新的、仅包含当前Program前向内容的Program。否则返回一个新的，和当前Program完全相同的Program

返回类型： Program

**代码示例**

注意，Program在clone后的顺序可能不同，这不会影响的训练或测试进程。在下面的示例中，我们提供了一个简单的方法print_prog（Program）来打印程序描述，以确保clone后仍能得到同样的打印结果：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

1.克隆一个Program，示例代码如下。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

2.如果分别运行 train Program 和 test Program，则可以不使用clone。

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

上边两个代码片段生成和打印的Program是一样的。

.. py:staticmethod:: parse_from_string(binary_str)

通过对 `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_ 的反序列化，转换成Program


参数：
 - **binary_str_type** (str) – `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_ 二进制字符串

返回：反序列化后的 Program

返回类型：Program

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:attribute:: num_blocks

该Program中的 :ref:`api_guide_Block` 的个数

返回： 该Program中的 :ref:`api_guide_Block` 的个数

返回类型：int

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:attribute:: random_seed

**注意：必须在相关OP被添加之前设置。**

程序中随机运算符的默认随机种子。0意味着随机生成随机种子。

返回：该Program中当前正在使用的random seed

返回类型：int64

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: global_block()

获取该Program的第一个 :ref:`api_guide_Block` 。

返回：该Program的第一个 :ref:`api_guide_Block`

返回类型：:ref:`api_guide_Block`

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: block(index)

返回该Program中 ， ``index`` 指定的 :ref:`api_guide_Block` 。 ``index`` 类型为int

参数:
 - **index** (int) - 需要获取的 :ref:`api_guide_Block`  的index

返回: 该Program中index对应的那个 :ref:`api_guide_Block`

返回类型: :ref:`api_guide_Block`

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: current_block()

获取当前 :ref:`api_guide_Block` 。当前 :ref:`api_guide_Block`  是用来添加OP的。

返回: 该Program中用户当前所在的 :ref:`api_guide_Block`

返回类型: :ref:`api_guide_Block`

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: list_vars()

获取当前Program中所有变量。返回值是一个可迭代对象（iterable object)。

返回: Generator 会yield每个Program中的变量

返回类型: iterable 的 :ref:`api_guide_Variable`


**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

.. py:method:: all_parameters()

获取当前Program中所有的 :ref:`api_guide_parameter` 。返回值是一个列表。

返回: 一个包含当前Program中所有参数的列表。

返回类型: list[ :ref:`api_guide_parameter` ]


**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    main_program = paddle.Program()
    startup_program = paddle.Program()
    with paddle.program_guard(main_program=main_program, startup_program=
        startup_program):
        x = fluid.layers.data(name='x', shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name='y', shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name='fc', input=x, size=10, act='relu')
    
    # start_up program here will share fc's weight with main program
    print('main program is: {}'.format(main_program))
    print('start up program is: {}'.format(startup_program))

