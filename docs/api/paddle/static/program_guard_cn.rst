.. _cn_api_fluid_program_guard:

program_guard
-------------------------------


.. py:function:: paddle.static.program_guard(main_program, startup_program=None)




配合使用 python 的 ``with`` 语句来将 ``with`` block 里的算子和变量添加进指定的全局主程序（main program）和启动程序（startup program）。

``with`` 语句块下的各接口将在新的 main program（主程序）中添加 operators（算子）和 Tensors。

参数
::::::::::::

    - **main_program** (Program) – ``with`` 语句中将使用的新的 main program。
    - **startup_program** (Program，可选) – ``with`` 语句中将使用的新的 startup program。若传入 ``None`` 则不改变当前的启动程序，即仍使用 default_startup_program。默认值为 None。

代码示例 1
::::::::::::

.. code-block:: python

    import paddle

    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
        hidden = paddle.static.nn.fc(x=data, size=10, activation='relu')

例如，当组的网不需要 startup_program 初始化各变量时，可以传入一个临时的 program。

代码示例 2
::::::::::::

.. code-block:: python

    import paddle

    paddle.enable_static()
    main_program = paddle.static.Program()
    # does not care about startup program. Just pass a temporary value.
    with paddle.static.program_guard(main_program, paddle.static.Program()):
        data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
