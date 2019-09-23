.. _cn_api_fluid_layers_less_than:

less_than
-------------------------------

.. py:function:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None)


该OP逐元素地返回 :math:`x < y` 的逻辑值，使用重载算子 `<` 可以有相同的计算函数效果


参数：
    - **x** (Variable) - 进行比较的第一个输入，是一个多维的Tensor，数据类型可以是float32，float64，int32，int64。
    - **y** (Variable) - 进行比较的第二个输入，是一个多维的Tensor，数据类型可以是float32，float64，int32，int64。
    - **force_cpu** (bool) – 值为True则强制将输出变量写入CPU内存中，否则将其写入目前所在的运算设备上。默认为True。
    - **cond** (Variable，可选) – 指定算子输出结果的Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。


返回：输出结果的Tensor，数据的shape和输入x一致。

返回类型： Variable，数据类型为bool。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    import numpy as np
    x = fluid.layers.data(name='x', shape=[1], dtype='float64')
    y = fluid.layers.data(name='y', shape=[1], dtype='float64')
    result = fluid.layers.less_than(x=x, y=y)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x_i = np.array([[1, 2], [3, 4]]).astype(np.float64)
    y_i = np.array([[2, 2], [1, 3]]).astype(np.float64)
    result_value, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[result])
    print(result_value) # [[True, False], [False, False]]

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    import numpy as np
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    result = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0)
    fluid.layers.less_than(x=x, y=y, cond=result)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x_i = np.array([[1, 2], [3, 4]]).astype(np.float64)
    y_i = np.array([[2, 2], [1, 3]]).astype(np.float64)
    result_value, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[result])
    print(result_value) # [[True, False], [False, False]]

