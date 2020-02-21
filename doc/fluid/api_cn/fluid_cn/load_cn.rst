.. _cn_api_fluid_load:

load
-------------------------------

.. py:function:: paddle.fluid.load(program, model_path, executor=None, var_list=None)

该接口从Program中过滤出参数和优化器信息，然后从文件中获取相应的值。

如果Program和加载的文件之间参数的维度或数据类型不匹配，将引发异常。

该函数还可以加载用[save_params，save_persistables，save_vars]接口保存的模型文件。
加载单个模型文件时，var_list不为None（当调用save_params、save_persistables或save_vars接口时文件名不为None）。

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要加载的Program。
 - **model_path**  (str) – 保存program的文件前缀。格式为 ``目录名称/文件前缀`` 。
 - **executor** (Executor, 可选) - 当startup program没有运行时，用于初始化参数的Executor。默认值：None。
 - **var_list** (list, 可选) - 加载使用[save_params，save_persistables，save_vars]接口保存的单个模型文件的变量列表。默认值：None。

返回: 无

**代码示例**

.. code-block:: python

    # example1
    import paddle.fluid as fluid

    x = fluid.data( name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc(x, 10)
    z = fluid.layers.fc(y, 10)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.save(fluid.default_main_program(), "./test_path")
    fluid.load(fluid.default_main_program(), "./test_path")

    # example2，注意example1和example2应该分开执行，避免干扰。
    # 执行example2时，确保之前调用fluid.save存储过相关信息。
    import paddle.fluid as fluid
    x = fluid.data( name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc(x, 10)
    z = fluid.layers.fc(y, 10)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    fluid.load(fluid.default_main_program(), "./test_path", exe)

