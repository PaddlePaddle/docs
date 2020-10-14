.. _cn_api_fluid_default_startup_program:




default_startup_program
-------------------------------

.. py:function:: paddle.fluid.default_startup_program()






该函数可以获取默认/全局 startup :ref:`cn_api_fluid_Program` (初始化启动程序)。

``paddle.nn`` 中的函数将参数初始化OP追加到 ``startup program`` 中， 运行 ``startup program`` 会完成参数的初始化。

该函数将返回默认的或当前的 ``startup program`` 。用户可以使用 :ref:`cn_api_fluid_program_guard` 来切换 :ref:`cn_api_fluid_Program` 。

返回: 当前的默认/全局的 ``startup program`` 。

返回类型: :ref:`cn_api_fluid_Program`

**代码示例：**

.. code-block:: python

        import paddle
        
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            x = paddle.data(name="x", shape=[-1, 784], dtype='float32')
            y = paddle.data(name="y", shape=[-1, 1], dtype='int32')
            z = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
            print("main program is: {}".format(paddle.static.default_main_program()))
            print("start up program is: {}".format(paddle.static.default_startup_program()))



