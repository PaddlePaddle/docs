.. _cn_api_fluid_default_startup_program:




default_startup_program
-------------------------------

.. py:function:: paddle.static.default_startup_program()






该函数可以获取默认/全局 startup :ref:`cn_api_fluid_Program` (初始化启动程序)。

``paddle.nn`` 中的函数将参数初始化OP追加到 ``startup program`` 中， 运行 ``startup program`` 会完成参数的初始化。

该函数将返回默认的或当前的 ``startup program`` 。用户可以使用 :ref:`cn_api_fluid_program_guard` 来切换 :ref:`cn_api_fluid_Program` 。

返回
:::::::::
 :ref:`cn_api_fluid_Program` ，当前的默认/全局的 ``startup program`` 。


代码示例
:::::::::

.. code-block:: python

        import paddle

        paddle.enable_static()
        x = paddle.static.data(name="x", shape=[-1, 784], dtype='float32')
        out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
        print("main program is: {}".format(paddle.static.default_main_program()))
        print("start up program is: {}".format(paddle.static.default_startup_program()))
