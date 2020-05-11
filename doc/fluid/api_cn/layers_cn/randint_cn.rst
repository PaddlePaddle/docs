.. _cn_api_fluid_layers_randint:

randint
-------------------------------

.. py:function:: paddle.fluid.layers.randint(low, high=None, shape=None, out=None, dtype=None, device=None, stop_gradient=False, seed=0, name=None)

该OP使用从区间[low，high)内均匀分布采样的随机整数初始化一个Tensor。当high为None时（默认），均匀采样的区间为[0,low)。

参数：
    - **low** (int)-要生成的随机值范围的下限，low包含在范围中。当high为None时，均匀采样的区间为[0,low)。
    - **high** (int，可选)-要生成的随机值范围的上限，high不包含在范围中。默认值为None。
    - **shape** (list|tuple|Variable，可选)-输出Tensor的维度，shape类型支持list，tuple，Variable。如果shape类型是list或者tuple，它的元素可以是整数或者形状为[1]的Tensor，其中整数的数据类型为int，Tensor的数据类型为int32或int64。如果shape的类型是Variable，则是1D的Tensor，Tensor的数据类型为int32或int64。如果shape为None，则会将shape设置为[1]。默认值为None。
    - **out** (Variable，可选)-用于存储创建的Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选)- 输出Tensor的数据类型，支持数据类型为int32，int64。如果dtype为None，则会将dtype设置为int64。默认值为None。
    - **device** (str， 可选)-指定在GPU或CPU上创建Tensor。如果device为None，则将选择运行Paddle程序的设备，默认为None。
    - **stop_gradient** (bool，可选)-指定是否停止梯度计算，默认值为False。
    - **seed** (int，可选)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。默认为 0。
    - **name** (str，可选)-具体用法请参见:ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：表示一个随机初始化结果的Tensor，该Tensor的数据类型由dtype参数决定，该Tensor的维度由shape参数决定。

返回类型：Variable

抛出异常：
    - :code:`TypeError`: shape的类型应该是list、tuple 或 Variable。
    - :code:`TypeError`: dtype的类型应该是int32或int64。
    - :code:`ValueError`: 该OP的high必须大于low（high为None时，则会先将high设置为low，将low设置为0，再判断low和high的大小关系）。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    # example 1:
    # attr shape is a list which doesn't contain tensor Variable.
    result_1 = fluid.layers.randint(low=-5, high=5, shape=[3, 4], dtype="int64")

    # example 2:
    # attr shape is a list which contains tensor Variable.
    dim_1 = fluid.layers.fill_constant([1],"int64",3)
    dim_2 = fluid.layers.fill_constant([1],"int32",5)
    result_2 = fluid.layers.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")

    # example 3:
    # attr shape is a Variable, the data type must be int64 or int32.
    var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
    result_3 = fluid.layers.randint(low=-5, high=5, shape=var_shape, dtype="int32")
    var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
    result_4 = fluid.layers.randint(low=-5, high=5, shape=var_shape_int32, dtype="int64")

    # example 4:
    # Input only one parameter
    # low=0, high=10, shape=[1], dtype='int64'
    result_4 = fluid.layers.randint(10)
