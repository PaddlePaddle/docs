.. _cn_api_fluid_layers_While:

While
-------------------------------


.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)





该类用于实现while循环控制功能，只要循环条件cond为True，就循环执行while循环体中的语句，直到cond为False为止。

.. note::
    如果参数 ``cond`` 的形状为[1]，强烈建议您使用新的OP :ref:`cn_api_fluid_layers_while_loop` 而不是 ``While``。
    OP :ref:`cn_api_fluid_layers_while_loop` 的使用方式更简单，并且调用该OP所用的代码更少且功能与 ``While`` 一样。

**注意：**
    在 ``While`` 中创建的局部变量类似于C++中的while，无法被外部引用，因此无法通过 ``Executor`` 中的 ``fetch_list`` 来获取。
    若想实现该功能，PaddlePaddle提供了 ``assign`` 接口将局部变量赋值到外部，请参考示例代码2 或参考 `issue#22724 <https://github.com/PaddlePaddle/Paddle/issues/22724>`_ 。

参数
::::::::::::

    - **cond** (Variable) – 用于判断循环继续进行的条件，为数据类型bool型的Tensor，其shape必须为[1]。
    - **is_test** (bool，可选) – 用于表明是否在测试阶段执行，默认值为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例 1
::::::::::::

.. code-block:: python

    # 该示例代码展示整数循环+1，循环10次，输出计数结果
    import paddle.fluid as fluid
    import numpy as np

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)           # 循环计数器
    
    loop_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=10)    # 循环次数

    cond = fluid.layers.less_than(x=i, y=loop_len)              # 循环条件   
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():  # 循环体
        i = fluid.layers.increment(x=i, value=1, in_place=True)
        fluid.layers.less_than(x=i, y=loop_len, cond=cond)      # 更新循环条件

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])
    print(res) # [array([10])]


代码示例 2
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    loop_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    one = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1)
    data = fluid.data(name='data', shape=[1], dtype='float32')
    sums = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0)  # 在 While 外先定义要获取的变量，需和要获取的 While 内部的变量名称不同

    cond = fluid.layers.less_than(x=i, y=loop_len)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        sums_tensor = fluid.layers.elementwise_add(x=data, y=data)
        fluid.layers.assign(input=sums_tensor, output=sums)  # 将 While 内定义的变量 sums_tenosr 通过 layers.assign 更新至 While 外的变量 sums 中
        i = fluid.layers.increment(x=i, value=1, in_place=True)
        data = fluid.layers.elementwise_add(x=data, y=one)
        fluid.layers.less_than(x=i, y=loop_len, cond=cond)

    feed_data = np.ones([1]).astype('float32')
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    res = exe.run(fluid.default_main_program(), feed={'data': feed_data}, fetch_list=sums)
    print(res[0])  # [2.]    # 因 While 内的 data 没有将值更新到 While 外，故循环过后此处 sums 的值为 [2.]









