.. _cn_api_fluid_save:

save
-------------------------------


.. py:function:: paddle.fluid.save(program, model_path)

:api_attr: 声明式编程模式（静态图)
:alias_main: paddle.static.save
:alias: paddle.static.save
:old_api: paddle.fluid.save



该接口将传入的参数、优化器信息和网络描述保存到 ``model_path`` 。

参数包含所有的可训练 :ref:`cn_api_fluid_Variable` ，将保存到后缀为 ``.pdparams`` 的文件中。

优化器信息包含优化器使用的所有变量。对于Adam优化器，包含beta1、beta2、momentum等。
所有信息将保存到后缀为 ``.pdopt`` 的文件中。（如果优化器没有需要保存的变量（如sgd），则不会生成）。

网络描述是程序的描述。它只用于部署。描述将保存到后缀为 ``.pdmodel`` 的文件中。

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要保存的Program。
 - **model_path**  (str) – 保存program的文件前缀。格式为 ``目录名称/文件前缀``。如果文件前缀为空字符串，会引发异常。

返回: 无

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    paddle.enable_static()

    x = fluid.data(name="x", shape=[10, 10], dtype='float32')
    y = fluid.layers.fc(x, 10)
    z = fluid.layers.fc(y, 10)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.save(fluid.default_main_program(), "./test_path")







