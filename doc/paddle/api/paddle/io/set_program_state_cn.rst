.. _cn_api_fluid_io_set_program_state:

set_program_state
-------------------------------

.. py:function:: paddle.fluid.io.set_program_state(program, state_dict)

:api_attr: 声明式编程模式（静态图)



利用 ``state_dict`` 设置 ``Program`` 的参数和优化器信息。

如果参数的 shape 或 dtype 不匹配，则会引发异常。

**注意：必须在运行 start_up_program 之后调用此函数。**

参数:
    - **program** (Program) - 需要被设置的 ``Program`` 。
    - **state_dict** (dict) - 存储参数和优化器信息的dict；dict中key的类型为变量的名称，value为np.ndarray类型的数据。

返回：无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.data( name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc( x, 10)
    z = fluid.layers.fc( y, 10)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run( fluid.default_startup_program() )
    prog = fluid.default_main_program()

    fluid.save( prog, "./temp")
    program_state = fluid.load_program_state( "./temp")
    fluid.set_program_state( prog, program_state)

