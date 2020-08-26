.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)

:api_attr: 声明式编程模式（静态图)



Sequence Expand As Layer，该OP根据输入 ``y`` 的第0级lod对输入 ``x`` 进行扩展。当前实现要求 ``y`` 的lod层数（level）必须为1，且 ``x`` 的第一维必须和 ``y`` 的第0层lod大小相同，所以扩展后的LodTensor具有和 ``y`` 相同的lod。扩展结果与输入 ``x`` 的lod无关，所以无需考虑 ``x`` 的lod。

注意，该OP的输入 ``x`` 可以是Tensor或LoDTensor， ``y`` 只能是LodTensor。

范例解释如下：

::

    例1:
    假设，有4个长度维1的序列[a]、[b]、[c]和[d]，现在要将其扩展为长度是3、3、1、1的序列[a][a][a]、[b][b][b]、[c]和[d]。
    显然，扩展后的序列lod为[0, 3, 6, 7, 8]，则：
    给定输入一维LoDTensor x
        x.data = [[a], [b], [c], [d]]
        x.dims = [4, 1]
    和输入 y
        y.lod = [[3, 3, 1, 1]]    #为了便于理解这里用基于长度lod表示
    
    经过sequence_expand_as运算，得到输出1级LoDTensor out
        out.lod =  [[0,            3,              6,  7,  8]]    #基于偏移的lod，等价于基于长度的[[3, 3, 1, 1]]
        out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        out.dims = [8, 1]
    
    可见，输出out将x扩展至和y具有相同的lod。

::

    例2：
    设定与例1类似，给定输入一维LoDTensor x：
        x.data = [[a, b], [c, d], [e, f]]
        x.dims = [3, 2]
    和输入 y：
        y.lod = [[2, 1, 3]]    #为了便于理解这里用基于长度lod表示

    输出为1级LoDTensor：
        out.lod =  [[0,             2,     3,                    6]]    #基于偏移的lod，等价于基于长度的[[2, 1, 3]]
        out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
        out.dims = [6, 2]

    可见，输出out将x扩展至和y具有相同的lod。


参数：
    - **x** (Variable) - 输入变量，维度为 :math:`[M, K]` 的二维Tensor或LoDTensor，第一维必须与输入 ``y`` 的第0层lod大小相同，且仅支持lod_level为1。数据类型支持int32，int64，float32或float64。
    - **y** (Variable) - 输入变量，LoDTensor，lod level必须为1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：扩展变量，维度为 :math:`[N, K]` 的二维LoDTensor，N由输入 ``y`` 的lod决定，且仅支持lod_level为1。数据类型与输入 ``x`` 一致。

返回类型：Variable


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    import numpy as np

    x = fluid.data(name='x', shape=[1], dtype='float32')
    y = fluid.data(name='y', shape=[1], dtype='float32', lod_level=1)
    out = layers.sequence_expand_as(x=x, y=y)

    exe = fluid.Executor(fluid.CPUPlace())
    place = fluid.CPUPlace()

    np_data = np.array([[1], [2], [3], [4]]).astype('float32')
    x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
    print(x_lod_tensor)
    #lod: [[0, 2, 4]]
    #    dim: 4, 1
    #    layout: NCHW
    #    dtype: float
    #    data: [1 2 3 4]

    y_lod_tensor = fluid.create_random_int_lodtensor([[3,3,1,1]], [1], 
                                                    place, low=0, high=1)
    print(y_lod_tensor)
    #lod: [[0, 3, 6, 7, 8]]
    #    dim: 8, 1
    #    layout: NCHW
    #    dtype: int64_t
    #    data: [0 0 1 0 1 1 1 0]

    out_main = exe.run(fluid.default_main_program(), 
                      feed={'x': x_lod_tensor, 'y': y_lod_tensor}, 
                      fetch_list=[out], return_numpy=False)
    print(out_main[0])
    #lod: [[0, 3, 6, 7, 8]]
    #    dim: 8, 1
    #    layout: NCHW
    #    dtype: float
    #    data: [1 1 1 2 2 2 3 4]









