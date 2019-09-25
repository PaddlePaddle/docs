.. _cn_api_fluid_layers_uniform_random:

uniform_random
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0)

该OP使用从均匀分布中抽样的随机值初始化一个Tensor。

::

    示例1:
             给定：
                 shape=[1,2]
             则输出为：
                 result=[[0.8505902, 0.8397286]]

参数：
    - **shape** (list|tuple|Variable)-输出Tensor的维度，数据类型是int64，如果shape类型是list或者tuple，它的元素可以是整数或者形状为[1]的Tensor。如果shape的类型是Variable，则是1D的Tensor。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) – 输出tensor的数据类型，支持float32（默认）， float64。
    - **min** (float，可选）-均匀随机分布的最小值，为闭区间。数据类型为float。默认值为-1.0。
    - **max** (float，可选)-均匀随机分布的最大值，为开区间。数据类型为floa。默认值为1.0。
    - **seed** (int，可选)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。数据类型为int。默认为 0。

返回：表示一个随机初始化结果的Tensor，该Tensor的数据类型由dtype参数决定，该Tensor的维度由shape参数决定。
    
返回类型：Variable

抛出异常：
    - :code:`ValueError`: shape的类型应该是list、tuple 或 Variable。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    startup_program = fluid.Program()
    train_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        # example 1:
        # attr shape is a list which doesn't contain tensor Variable.
        result_1 = fluid.layers.uniform_random(shape=[3, 4])

        # example 2:
        # attr shape is a list which contains tensor Variable.
        dim_1 = fluid.layers.fill_constant([1],"int64",3)
        result_2 = fluid.layers.uniform_random(shape=[dim_1, 5])

        # example 3:
        # attr shape is a Variable, the data type must be int64
        var_shape = fluid.layers.data(name='var_shape',shape=[2],append_batch_size=False)
        result_3 = fluid.layers.uniform_random(var_shape)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)
        outs = exe.run(train_program, feed = {'var_shape':np.array([3,4])}, fetch_list=[result_1, result_2, result_3])







