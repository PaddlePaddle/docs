.. _cn_api_fluid_program_guard:

program_guard
-------------------------------


.. py:function:: paddle.fluid.program_guard(main_program, startup_program=None)

:api_attr: 声明式编程模式（静态图)



该接口应配合使用python的 ``with`` 语句来将 ``with`` block 里的算子和变量添加进指定的全局主程序（main program）和启动程序（startup program）。

``with`` 语句块中的fluid.layers下各接口将在新的main program（主程序）中添加operators（算子）和variables（变量）。

参数：
    - **main_program** (Program) – “with”语句中将使用的新的main program。
    - **startup_program** (Program，可选) – “with”语句中将使用的新的startup program。若传入 ``None`` 则不改变当前的启动程序，即仍使用default_startup_program。默认值为None。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        data = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10, act='relu')

例如，当组的网不需要startup_program初始化各变量时，可以传入一个临时的program。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    main_program = fluid.Program()
    # 如果您不需要关心startup program,传入一个临时值即可
    with fluid.program_guard(main_program, fluid.Program()):
        data = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')

