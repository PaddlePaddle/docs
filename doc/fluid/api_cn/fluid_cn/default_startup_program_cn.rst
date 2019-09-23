.. _cn_api_fluid_default_startup_program:




default_startup_program
-------------------------------

.. py:function:: paddle.fluid.default_startup_program()



该函数可以获取默认/全局 startup :ref:`cn_api_fluid_Program` (初始化启动程序)。

 :ref:`_cn_api_fluid_layers` 中的函数会新建参数或 :ref:`cn_api_paddle_data_reader_reader` (读取器) 或 `NCCL <https://developer.nvidia.com/nccl>`_ 句柄作为全局变量。

startup_program会使用内在的OP（算子）去初始化他们，并由 :ref:`_cn_api_fluid_layers` 中的函数将这些OP追加到startup :ref:`cn_api_fluid_Program` 中。

该函数将返回默认的或当前的startup_program。用户可以使用 :ref:`cn_api_fluid_program_guard`  去切换 :ref:`cn_api_fluid_Program` 。

返回: 当前的默认/全局 初始化 :ref:`cn_api_fluid_Program`

返回类型: :ref:`cn_api_fluid_Program`

**代码示例：**

.. code-block:: python

        import paddle.fluid as fluid
     
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
            z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")
     
            print("main program is: {}".format(fluid.default_main_program()))
            print("start up program is: {}".format(fluid.default_startup_program()))



