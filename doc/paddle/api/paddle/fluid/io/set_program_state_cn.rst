.. _cn_api_fluid_io_set_program_state:

set_program_state
-------------------------------

.. py:function:: paddle.static.set_program_state(program, state_dict)


利用 ``state_dict`` 设置 ``Program`` 的参数和优化器信息。

如果参数的 shape 或 dtype 不匹配，则会引发异常。

**注意：必须在运行 start_up_program 之后调用此函数。**

参数:
    - **program** (Program) - 需要被设置的 ``Program`` 。
    - **state_dict** (dict) - 存储参数和优化器信息的dict；dict中key的类型为Tensor的名称，value为np.ndarray类型的数据。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    import paddle.static as static

    paddle.enable_static()

    x = static.data(name="x", shape=[10, 10], dtype='float32')
    y = static.nn.fc(x, 10)
    z = static.nn.fc(y, 10)

    place = paddle.CPUPlace()
    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()

    static.save(prog, "./temp")
    program_state = static.load_program_state("./temp")

    static.set_program_state(prog, program_state)

