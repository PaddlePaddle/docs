.. _cn_api_fluid_io_load_program_state:

load_program_state
-------------------------------

.. py:function:: paddle.fluid.io.load_program_state(model_path, var_list=None)

该接口从本地文件中加载 ``Program`` 的状态。

参数:
    - **model_path** (str) - 存储 ``Program`` 信息文件的前缀
    - **var_list** (list, optional) - 加载的变量列表（使用[save_params，save_persistables，save_vars]接口保存的变量）

返回：存储参数和优化器信息的dict

返回类型：dict

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

