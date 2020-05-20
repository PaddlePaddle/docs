.. _cn_api_fluid_layers_While:

While
-------------------------------


.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)

:api_attr: 声明式编程模式（静态图)




该类用于实现while循环控制功能，只要循环条件cond为True，就循环执行while循环体中的语句，直到cond为False为止。

.. note::
    如果参数 ``cond`` 的形状为[1]，强烈建议您使用新的OP :ref:`cn_api_fluid_layers_while_loop` 而不是 ``While``。
    OP :ref:`cn_api_fluid_layers_while_loop` 的使用方式更简单，并且调用该OP所用的代码更少且功能与 ``While`` 一样。

**注意：**
    在 ``While`` 中创建的局部变量类似于C++中的while，无法被外部引用，因此无法通过 ``Executor`` 中的 ``fetch_list`` 来获取。
    若想实现该功能，PaddlePaddle提供了 ``assign`` 接口将局部变量赋值到外部，请参考示例代码2 或参考 `issue#22724 <https://github.com/PaddlePaddle/Paddle/issues/22724>`_ 。

参数：
    - **cond** (Variable) – 用于判断循环继续进行的条件，为数据类型bool型的Tensor，其shape必须为[1]。
    - **is_test** (bool，可选) – 用于表明是否在测试阶段执行，默认值为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

**代码示例 1**

.. code-block:: python

    # 该示例代码展示整数循环+1，循环10次，输出计数结果
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    i = paddle.full(shape=[1], dtype='int64', fill_value=0, device=None,
    
        stop_gradient=True)
    loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10, device=None,
    
        stop_gradient=True)
    cond = paddle.less_than(x=i, y=loop_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        i = paddle.increment(x=i, value=1, in_place=True)
    
        paddle.less_than(x=i, y=loop_len, cond=cond)
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    res = exe.run(paddle.default_main_program(), feed={}, fetch_list=[i])
    print(res)

**代码示例 2**

.. code-block:: python

    # 该示例代码展示整数循环+1，循环10次，输出计数结果
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    i = paddle.full(shape=[1], dtype='int64', fill_value=0, device=None,
    
        stop_gradient=True)
    loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10, device=None,
    
        stop_gradient=True)
    cond = paddle.less_than(x=i, y=loop_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        i = paddle.increment(x=i, value=1, in_place=True)
    
        paddle.less_than(x=i, y=loop_len, cond=cond)
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    res = exe.run(paddle.default_main_program(), feed={}, fetch_list=[i])
    print(res)

