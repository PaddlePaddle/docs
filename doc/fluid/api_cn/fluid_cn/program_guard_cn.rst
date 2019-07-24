.. _cn_api_fluid_program_guard:

program_guard
-------------------------------

.. py:function::    paddle.fluid.program_guard(main_program, startup_program=None)



该函数应配合使用python的“with”语句来改变全局主程序(main program)和启动程序(startup program)。

“with”语句块中的layer函数将在新的main program（主程序）中添加operators（算子）和variables（变量）。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        data = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10, act='relu')

需要注意的是，如果用户不需要构建自己的启动程序或者主程序，一个临时的program将会发挥作用。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    # 如果您不需要关心startup program,传入一个临时值即可
    with fluid.program_guard(main_program, fluid.Program()):
        data = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')


参数：
    - **main_program** (Program) – “with”语句中将使用的新的main program。
    - **startup_program** (Program) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序。










