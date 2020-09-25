.. _cn_api_fluid_io_load_program_state:

load_program_state
-------------------------------

.. py:function:: paddle.fluid.io.load_program_state(model_path, var_list=None)




该接口从本地加载 ``Program`` 的参数和优化器的变量信息到内存中。

参数:
    - **model_path** (str) - 存储 ``Program`` 的参数和优化器的变量信息的目录名称+文件前缀，格式为 ``目录名称/文件前缀`` 。
    - **var_list** (list, 可选) - 指定加载的变量列表，该参数只在加载旧接口[save_params，save_persistables，save_vars]保存的模型文件时使用。当加载的是多个小文件时，变量列表可以是所有加载文件中变量的子集；当加载的单个大文件时，变量列表必须和加载文件中的变量保持一致。

返回：存储参数和优化器信息的dict

返回类型：dict

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    paddle.enable_static()

    x = fluid.data( name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc( x, 10)
    z = fluid.layers.fc( y, 10)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run( fluid.default_startup_program() )
    prog = fluid.default_main_program()

    fluid.save( prog, "./temp")
    program_state = fluid.load_program_state( "./temp")

