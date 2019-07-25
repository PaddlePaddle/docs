.. _cn_api_fluid_layers_sequence_scatter:

sequence_scatter
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_scatter(input, index, updates, name=None)

序列散射层

这个operator将更新张量X，它使用Ids的LoD信息来选择要更新的行，并使用Ids中的值作为列来更新X的每一行。

**样例**:

::

    输入：

    input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    input.dims = [3, 6]

    index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
    index.lod =  [[0,        3,                       8,                 12]]

    updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
    updates.lod =  [[  0,            3,                                 8,                         12]]

    输出：

    out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
    out.dims = X.dims = [3, 6]



参数：
      - **input** (Variable) - input 秩（rank） >= 1。
      - **index** (Variable) - LoD Tensor， index 是 sequence scatter op 的输入索引，该函数的input将依据index进行更新。 秩（rank）=1。由于用于索引dtype应该是int32或int64。
      - **updates** (Variable) - 一个 LoD Tensor , update 的值将被 sactter 到输入x。update 的 LoD信息必须与index一致。
      - **name** (str|None) - 输出变量名。默认：None。

返回： 输出张量维度应该和输入张量相同

返回类型：Variable


**代码示例**:

..  code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
     
    input = layers.data( name="x", shape=[3, 6], append_batch_size=False, dtype='float32' )
    index = layers.data( name='index', shape=[1], dtype='int32')
    updates = layers.data( name='updates', shape=[1], dtype='float32')
    output = fluid.layers.sequence_scatter(input, index, updates)










