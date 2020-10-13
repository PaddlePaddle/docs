.. _cn_api_fluid_dygraph_ProgramTranslator

ProgramTranslator
-------------------------------

.. py:class:: paddle.jit.ProgramTranslator()

将动态图函数转为静态图函数的类。该类是个单例（singleton）。

参数：
    无。

返回：ProgramTranslator 单例对象。

**示例代码**

.. code-block:: python

    import paddle

    # 以下两种调用方法得到同一个对象，因为ProgramTranslator是个单例
    paddle.jit.ProgramTranslator()
    paddle.jit.ProgramTranslator.get_instance()

.. py:method:: enable(enable_static)

全局开启或关闭动态图转化为静态图。

参数：
    - **enable_static** (bool) - 设置True或者False来打开或关闭动静转化 。

返回：None。

**示例代码**

.. code-block:: python

    import paddle


    @paddle.jit.to_static
    def func(x):
        if paddle.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v


    prog_trans = paddle.jit.ProgramTranslator()
    prog_trans.enable(False)

    x = paddle.ones([1, 2])
    # ProgramTranslator被关闭所以func会以动态图模式运行
    print(func(x).numpy())  # [[0. 0.]]

.. py:method:: get_output(dygraph_func, *args, **kwargs)

返回动态图函数输出的Tensor，但是该动态图函数的数值计算过程会被转化为静态图模式运行。

参数：
    - **dygraph_func** (callable) - 动态图函数。
    - **args, kwargs** - 动态图函数的输入。

返回：包含数值结果的Tensor或者Tensor的元组，是输入动态图函数的返回值。

**示例代码**

.. code-block:: python

    import paddle


    def func(x):
        if paddle.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v


    prog_trans = paddle.jit.ProgramTranslator()

    x = paddle.ones([1, 2])
    x_v = prog_trans.get_output(func, x)
    print(x_v.numpy())  # [[0. 0.]]

.. py:method:: get_func(dygraph_func)

返回一个可调用函数，该函数将输入动态图函数接口转化为静态图组网接口。组网接口不像动态图接口，其并不直接返回数据结果。用户需要自行处理对应的Program和Eexecutor。

参数：
    - **dygraph_func** (callable) - 动态图函数。

返回：将动态图接口转为静态图组网接口的可调用函数。

**示例代码**

.. code-block:: python

    import paddle


    def func(x):
        if paddle.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v


    prog_trans = paddle.jit.ProgramTranslator()
    static_func = prog_trans.get_func(func)
    print(callable(static_func)) # True

.. py:method:: get_program(dygraph_func, *args, **kwargs)

返回动态图函数转化后的静态图Program和输入输出Varaible。用户可以使用Executor来执行该Program。

参数：
    - **dygraph_func** (callable) - 动态图函数。
    - **args, kwargs** - 动态图函数的输入。

返回：元组(main_program, startup_program, inputs, outputs)
    main_program: 转化后的main program。
    startup_program: 转化后的startup program。
    inputs: 输入Tensor的列表，这些Tensor可以在执行去feed。
    outputs: 输出Tensor的列表，这些Tensor可以在运行时被fetch。

**示例代码**

.. code-block:: python

    import paddle


    def func(x):
        if paddle.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v


    prog_trans = paddle.jit.ProgramTranslator()

    x = paddle.ones([1, 2])
    main_prog, start_prog, inputs, outputs = prog_trans.get_program(func, x)
    print([i.name for i in inputs])
    # [u'generated_tensor_0'] 需要被feed的输入Tensor名字，对应x
    print([o.name for o in outputs])
    # [u'_generated_var_4'] 需要被fetch的输出Tensor名字，对应x_v

.. py:method:: get_code(dygraph_func)

返回动态图函数转化后的静态图代码字符串。

参数：
    - **dygraph_func** (callable) - 动态图函数。

返回：转化后的静态图代码字符串。

**示例代码**

.. code-block:: python

    import paddle


    def func(x):
        if paddle.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v


    prog_trans = paddle.jit.ProgramTranslator()    

    code = prog_trans.get_code(func)
    print(type(code)) # <class 'str'>


.. py:method:: get_program_cache()

返回ProgramCache单例。这个方法是PaddlePaddle开发者用来管理ProgramTranslator中的Program缓存，普通用户不需要使用这个方法。

返回：ProgramTranslator中的ProgramCache。

**示例代码**

.. code-block:: python

    import paddle

    prog_trans = paddle.jit.ProgramTranslator()
    prog_cache = prog_trans.get_program_cache()

