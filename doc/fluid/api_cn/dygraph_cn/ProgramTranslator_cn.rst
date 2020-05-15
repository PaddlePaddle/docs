.. _cn_api_fluid_dygraph_ProgramTranslator

ProgramTranslator
-------------------------------

.. py:class:: paddle.fluid.dygraph.dygraph_to_static.ProgramTranslator()

将动态图函数转为静态图函数的类。该类是个单例（singleton）。

参数：
    无。

返回：ProgramTranslator 单例对象。

返回类型：ProgramTranslator。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid

    # 以下两种调用方法得到同一个对象，因为ProgramTranslator是个单例
    fluid.dygraph.ProgramTranslator()
    fluid.dygraph.ProgramTranslator.get_instance()

.. py:method:: enable(enable_declarative)

全局开启或关闭动态图转化为静态图。

参数：
    - **enable_declarative** (bool) - 设置True或者False来打开或关闭declarative 。

返回：None。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    @fluid.dygraph.jit.declarative
    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    prog_trans = fluid.dygraph.ProgramTranslator()
    prog_trans.enable(False)

    x = np.ones([1, 2])
    # The declarative is disabled so the func is run in dygraph
    with fluid.dygraph.guard():
        print(func(x).numpy()) # [[2. 2.]]

.. py:method:: get_output(dygraph_func, *args, **kwargs)

返回动态图函数输出的VarBase，但是该动态图函数的数值计算过程会被转化为静态图模式运行。

参数：
    - **dygraph_func** (callable) - 动态图函数。
    - **args, kwargs** - 动态图函数的输入。

返回：包含数值结果的VarBase或者VarBase的元组，是输入动态图函数的返回值。

返回类型：VarBase或者VarBase的元组。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    prog_trans = fluid.dygraph.ProgramTranslator()

    with fluid.dygraph.guard():
        x = np.ones([1, 2])
        x_v = prog_trans.get_output(func, x)
        print(x_v.numpy()) # [[0. 0.]]

.. py:method:: get_func(dygraph_func)

返回一个可调用函数，该函数将输入动态图函数接口转化为静态图组网接口。组网接口不像动态图接口，其并不直接返回数据结果。用户需要自行处理对应的Program和Eexecutor。

参数：
    - **dygraph_func** (callable) - 动态图函数。

返回：将动态图接口转为静态图组网接口的可调用函数。

返回类型：可调用函数。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    prog_trans = fluid.dygraph.ProgramTranslator()

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
    inputs: 输入Variable的列表，这些Variable可以在执行去feed。
    outputs: 输出Variable的列表，这些Variable可以在运行时被fetch。

返回类型：类型为(Program, Program, list(Variable), list(Variable)) 的元组。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    prog_trans = fluid.dygraph.ProgramTranslator()

    x = np.ones([1, 2])
    main_prog, start_prog, inputs, outputs = prog_trans.get_program(func, x)
    print([i.name for i in inputs])
    # ['feed_0'] 需要被feed的输入Variable名字，对应x
    print([o.name for o in outputs])
    # ['_generated_var_4'] 需要被fetch的输出Variable名字，对应x_v

.. py:method:: get_code(dygraph_func)

返回动态图函数转化后的静态图代码字符串。

参数：
    - **dygraph_func** (callable) - 动态图函数。

返回：转化后的静态图代码字符串。

返回类型：str。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) > 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    prog_trans = fluid.dygraph.ProgramTranslator()

    code = prog_trans.get_code(func)
    print(type(code)) # <class 'str'>

.. py:method:: save_inference_model(dirname, feed=None, fetch=None)

将现有模型保存为预测模型。保存过程会裁剪main program，只保存和预测输入输出有关的部分，构建成新的Program，并将此Program和相关参数保存到指定dirname路径下，被保存的模型可以被 :ref:`cn_api_fluid_io_load_inference_model` 或者C++预测接口使用。

参数：
    - **dirname** (str) - 存储预测模型的目录。
    - **feed (list[int], 可选)** - 预测模型要保存的输入Variable的序号。如果为None，则动态图函数的所有输入变量将被保存。默认值为None。
    - **fetch (list[int], 可选)** - 预测模型要保存的输出Variable的序号。如果为None，则动态图函数的所有输出变量将被保存。默认值为None。

返回：None。

**示例代码**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import Linear
    from paddle.fluid.dygraph import declarative
    from paddle.fluid.dygraph import ProgramTranslator

    class SimpleNet(fluid.dygraph.Layer):
        def __init__(self, in_size, out_size):
            super(SimpleNet, self).__init__()
            self._linear = Linear(in_size, out_size)

        @declarative
        def forward(self, x):
            y = self._linear(x)
            z = self._linear(y)
            loss = fluid.layers.mean(z)
            return z, loss

    with fluid.dygraph.guard(fluid.CPUPlace()):
        net = SimpleNet(8, 8)
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
        x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            loss, out = net(x)
            loss.backward()
            adam.minimize(loss)
            net.clear_gradients()
    # 保存模型
    # 注意fetch=[0]意味着我们将序号为0的动态图return输出'z'作为预测输出
    prog_trans = ProgramTranslator()
    prog_trans.save_inference_model("./dy2stat_infer_model", fetch=[0])

    # 在这个例子中，预测模型会根据输出'z'进行裁剪。被裁剪后的Program 会被保
    # 存在"./dy2stat_infer_model" 文件夹，并且参数也会保存为同一个文件夹下
    # 不同文件。

.. py:method:: get_program_cache()

返回ProgramCache单例。这个方法是PaddlePaddle开发者用来管理ProgramTranslator中的Program缓存，普通用户不需要使用这个方法。

返回：ProgramTranslator中的ProgramCache。

返回类型：ProgramCache。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid

    prog_trans = fluid.dygraph.ProgramTranslator()
    prog_cache = prog_trans.get_program_cache()

