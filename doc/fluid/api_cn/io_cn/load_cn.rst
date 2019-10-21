.. _cn_api_fluid_io_load:

load
-------------------------------

.. py:function:: paddle.fluid.io.load(program, model_path)

该接口从Program中过滤出参数和优化器信息，然后从文件中获取相应的值。

如果Program和加载的文件之间参数的维度或数据类型不匹配，将引发异常。

**注意：此函数必须在运行启动程序（start_up_program）之后再调用。**

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要加载的Program。
 - **model_path**  (str) – 保存program的文件前缀。格式为 ``目录名称/文件前缀``。

返回: 无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.data( name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc(x, 10)
    z = fluid.layers.fc(y, 10)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.save(fluid.default_main_program(), "./test_path")

    fluid.load(fluid.default_main_program(), "./test_path")



