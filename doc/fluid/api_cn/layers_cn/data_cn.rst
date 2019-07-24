.. _cn_api_fluid_layers_data:

data
-------------------------------

.. py:function:: paddle.fluid.layers.data(name, shape, append_batch_size=True, dtype='float32', lod_level=0, type=VarType.LOD_TENSOR, stop_gradient=True)

数据层(Data Layer)

该功能接受输入数据，判断是否需要以minibatch方式返回数据，然后使用辅助函数创建全局变量。该全局变量可由计算图中的所有operator访问。

这个函数的所有输入变量都作为本地变量传递给LayerHelper构造函数。

请注意，paddle在编译期间仅使用shape来推断网络中以下变量的形状。在运行期间，paddle不会检查所需数据的形状是否与此函数中的形状设置相匹配。

参数：
    - **name** (str)-函数名或函数别名
    - **shape** (list)-声明维度信息的list。如果 ``append_batch_size`` 为True且内部没有维度值为-1，则应将其视为每个样本的形状。 否则，应将其视为batch数据的形状。
    - **append_batch_size** (bool)-

        1.如果为真，则在维度shape的开头插入-1。
        例如，如果shape=[1],则输出shape为[-1,1]。这对在运行期间设置不同的batch大小很有用。

        2.如果维度shape包含-1，比如shape=[-1,1]。
        append_batch_size会强制变为为False（表示无效），因为PaddlePaddle不能在shape上设置一个以上的未知数。

    - **dtype** (np.dtype|VarType|str)-数据类型：float32,float_16,int等
    - **type** (VarType)-输出类型。默认为LOD_TENSOR
    - **lod_level** (int)-LoD层。0表示输入数据不是一个序列
    - **stop_gradient** (bool)-布尔类型，提示是否应该停止计算梯度

返回：全局变量，可进行数据访问

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='x', shape=[784], dtype='float32')










