.. _cn_api_fluid_Program:

Program
-------------------------------

.. py:class::  paddle.fluid.Program


创建python program， 在paddleFluid内部会被转换为ProgramDesc描述语言，用来创建一段 c++ 程序。Program像容器一样，是一种自包含的程序语言。Program中包括至少一个块（Block），当 block 中存在条件选择的控制流op（例如 while_op）时，该Program将会含有嵌套块（nested block）。详情请参阅framework.proto。

注意：默认情况下，paddleFluid内部默认含有 ``default_startup_program`` 和 ``default_main_program`` ，它们将共享参数。 ``default_startup_program`` 只运行一次来初始化参数， ``default_main_program`` 在每个mini batch中运行并调整权重。

返回： empty program

**代码示例**

.. code-block:: python
  
    import paddle.fluid as fluid

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
        y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
        z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")

    print("main program is: {}".format(main_program))
      
    print("start up program is: {}".format(startup_program))


.. py:method:: to_string(throw_on_error, with_details=False)

用于debug

参数：
  - **throw_on_error** (bool): 没有设置任何必需的字段时，抛出值错误。
  - **with_details** (bool): 值为true时，打印更多关于变量和参数的信息，如trainable, optimize_attr等

返回：(str): debug 字符串

返回类型： str

抛出异常：
 - ``ValueError`` - 当 ``throw_on_error == true`` ，但没有设置任何必需的字段时，抛出 ``ValueError`` 。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            prog_string = prog.to_string(throw_on_error=True, with_details=False)
            print(prog_string)

.. py:method:: clone(for_test=False)

创建一个新的、相同的Program。

有些operator，在训练和测试之间的行为是不同的，比如 ``batch_norm`` 。它们有一个属性 ``is_test`` 来控制行为。当 ``for_test=True`` 时，此方法将把它们的 ``is_test`` 属性更改为True。

- 克隆Program用于训练时，将 ``for_test`` 设置为False。
- 克隆Program用于测试时，将 ``for_test`` 设置为True。我们不会在此处对程序进行任何裁剪，因此，如果您只是想要一个用于测试的前向计算程序，请在使用 ``Opimizer.minimize`` 之前使用 ``clone``

注意: 
    1. ``Program.clone()`` 方法不会克隆 ``py_reader`` 
    2. 此API不会裁剪任何算子。请在backward和optimization之前使用 ``clone(for_test=True)`` 。例如：

    .. code-block:: python

          import paddle.fluid as fluid
          test_program = fluid.default_main_program().clone(for_test=True)
          optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
          optimizer.minimize()

参数：
  - **for_test** (bool) – 取值为True时，clone方法内部会把operator的属性 ``is_test`` 设置为 True

返回：一个新的、相同的Program

返回类型：Program

**代码示例**

注意，Program Desc在clone后的顺序可能不同，这不会影响您的训练或测试进程。在下面的示例中，我们为您提供了一个简单的方法print_prog（program）来打印程序描述，以确保clone后您仍能得到同样的打印结果：

.. code-block:: python     
                
        import paddle.fluid as fluid
        import six


        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))

1.克隆一个Program，示例代码如下。

.. code-block:: python

        import paddle.fluid as fluid
        import six

        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                img = fluid.layers.data(name='image', shape=[784])
                hidden = fluid.layers.fc(input=img, size=200, act='relu')
                hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
                loss = fluid.layers.cross_entropy(
                                          input=fluid.layers.fc(hidden, size=10, act='softmax'),
                            label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
                avg_loss = fluid.layers.mean(loss)
                test_program = train_program.clone(for_test=False)
        print_prog(test_program)
        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(avg_loss)
  
2.如果分别运行 train Program 和 test Program，则可以不使用clone。

.. code-block:: python

        import paddle.fluid as fluid
        import six

        def print_prog(prog):
            for name, value in sorted(six.iteritems(prog.block(0).vars)):
                print(value)
            for op in prog.block(0).ops:
                print("op type is {}".format(op.type))
                print("op inputs are {}".format(op.input_arg_names))
                print("op outputs are {}".format(op.output_arg_names))
                for key, value in sorted(six.iteritems(op.all_attrs())):
                    if key not in ['op_callstack', 'op_role_var']:
                        print(" [ attrs: {}:   {} ]".format(key, value))
        def network(is_test):
            img = fluid.layers.data(name='image', shape=[784])
            hidden = fluid.layers.fc(input=img, size=200, act='relu')
            hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
            loss = fluid.layers.cross_entropy(
                input=fluid.layers.fc(hidden, size=10, act='softmax'),
                label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
            avg_loss = fluid.layers.mean(loss)
            return avg_loss


        train_program_2 = fluid.Program()
        startup_program_2 = fluid.Program()
        test_program_2 = fluid.Program()
        with fluid.program_guard(train_program_2, startup_program_2):
            with fluid.unique_name.guard():
                 sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                 sgd.minimize(avg_loss)
        # 不使用测试阶段的启动程序
        with fluid.program_guard(test_program_2, fluid.Program()):
            with fluid.unique_name.guard():
                loss = network(is_test=True)
        print(test_program_2)

上边两个代码片段生成和打印的Program是一样的。

.. py:staticmethod:: parse_from_string(binary_str)

反序列化protobuf，转换成program

注意:在序列化和反序列化之后，所有关于参数的信息都会丢失。

参数:
    - **binary_str_type** (str) – prootbuf二进制字符串

返回: 反序列化后的ProgramDesc

返回类型：Program

.. py:attribute:: num_blocks

该program中的block的个数

**代码示例**

.. code-block:: python
            
            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            num_blocks = prog.num_blocks
            print(num_blocks)

.. py:attribute:: random_seed


程序中随机运算符的默认随机种子。0意味着从随机设备中获取随机种子。

注意：必须在operator被添加之前设置。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            random_seed = prog.random_seed
            print(random_seed)
            prog.random_seed = 1
            print(prog.random_seed)

.. py:method:: global_block()

获取该program的第一个block。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            gb_block = prog.global_block()
            print(gb_block)

.. py:method:: block(index)

返回该program中 ， ``index`` 指定的block。 ``index`` 类型为int

返回：index对应的block

返回类型：Block

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            block_0 = prog.block(0)
            print(block_0)

.. py:method:: current_block()

获取当前block。当前block是用来添加operators。

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            current_blk = prog.current_block()
            print(current_blk)

.. py:method:: list_vars()

获取当前program中所有变量。返回值是一个可迭代对象（iterable object)。

返回：generator 会yield每个Program中的变量

返回类型：iterable
  
**代码示例**

.. code-block:: python

            import paddle.fluid as fluid
     
            prog = fluid.default_main_program()
            img = fluid.layers.data(name='img', shape=[1,28,28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[128,1], dtype='int64')
            for var in prog.list_vars():
                print(var)




