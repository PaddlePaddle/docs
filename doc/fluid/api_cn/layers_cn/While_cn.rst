.. _cn_api_fluid_layers_While:

While
-------------------------------

.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)


该类用于实现while循环控制功能，只要循环条件cond为True，就循环执行while循环体中的语句，直到cond为False为止。


参数：
    - **cond** (Variable) – 用于判断循环继续进行的条件，为数据类型bool型的Tensor，其shape必须为[1]。
    - **is_test** (bool，可选) – 用于表明是否在测试阶段执行，默认值为False。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**代码示例**

..  code-block:: python

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












