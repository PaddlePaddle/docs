.. _cn_api_fluid_save:

save
-------------------------------

.. py:function:: paddle.static.save(program, model_path)

:api_attr: 声明式编程模式（静态图)

该接口将传入的参数、优化器信息和网络描述保存到 ``model_path`` 。

参数包含所有的可训练 :ref:`cn_api_fluid_Variable` ，将保存到后缀为 ``.pdparams`` 的文件中。

优化器信息包含优化器使用的所有Tensor。对于Adam优化器，包含beta1、beta2、momentum等。
所有信息将保存到后缀为 ``.pdopt`` 的文件中。（如果优化器没有需要保存的Tensor（如sgd），则不会生成）。

网络描述是程序的描述。它只用于部署。描述将保存到后缀为 ``.pdmodel`` 的文件中。

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要保存的Program。
 - **model_path**  (str) – 保存program的文件前缀。格式为 ``目录名称/文件前缀``。如果文件前缀为空字符串，会引发异常。

返回: 无

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
