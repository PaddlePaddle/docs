
.. _cn_api_paddle_tensor_meshgrid:

meshgrid
-------------------------------

.. py:function:: paddle.tensor.meshgrid(input, name=None)

:alias_main: paddle.meshgrid
:alias: paddle.meshgrid,paddle.tensor.meshgrid,paddle.tensor.creation.meshgrid



该OP的输入是tensor list, 包含 k 个一维Tensor，对每个Tensor做扩充操作，输出 k 个 k 维tensor。

参数：
         - **input** （Variable）- 输入变量为 k 个一维Tensor，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为float32，float64，int32，int64。
         - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 
k 个 k 维Tensor，每个Tensor的形状均为(N1, N2, ..., Nk)。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

    #静态图示例
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    x = paddle.data(name='x', shape=[100], dtype='int32')
    y = paddle.data(name='y', shape=[200], dtype='int32')
    input_1 = np.random.randint(0, 100, [100]).astype('int32')
    input_2 = np.random.randint(0, 100, [200]).astype('int32')
    exe = paddle.Executor(place=paddle.CPUPlace())
    grid_x, grid_y = paddle.tensor.meshgrid([x, y])
    res_1, res_2 = exe.run(paddle.default_main_program(), feed={'x': input_1,
        'y': input_2}, fetch_list=[grid_x, grid_y])

..  code-block:: python

    #静态图示例
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    x = paddle.data(name='x', shape=[100], dtype='int32')
    y = paddle.data(name='y', shape=[200], dtype='int32')
    input_1 = np.random.randint(0, 100, [100]).astype('int32')
    input_2 = np.random.randint(0, 100, [200]).astype('int32')
    exe = paddle.Executor(place=paddle.CPUPlace())
    grid_x, grid_y = paddle.tensor.meshgrid([x, y])
    res_1, res_2 = exe.run(paddle.default_main_program(), feed={'x': input_1,
        'y': input_2}, fetch_list=[grid_x, grid_y])

