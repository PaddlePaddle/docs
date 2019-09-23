.. _cn_api_fluid_embedding:

embedding
-------------------------------

.. py:function:: paddle.fluid.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

**注意：相对于 fluid.layers.** :ref:`cn_api_fluid_layers_embedding` **（此OP将在未来的版本中被弃用！），此OP移除了输入Tensor shape的最后一维强制为1的限制，详细使用区别，请参考使用代码样例。**

该OP根据input中的id信息从embedding矩阵中查询对应embedding信息，函数会根据输入的size (vocab_size, emb_size)和dtype自动构造一个二维embedding矩阵。

输出的Tensor的shape是在输入Tensor shape的最后一维后面添加了emb_size的维度。

注：input中的id必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


::

    Case 1:

    input是Tensor, 且padding_idx = -1
        input.data = [[1, 3], [2, 4], [4, 127]]
        input.shape = [3, 2]
    若size = [128, 16]
    输出为Tensor:
        out.shape = [3, 2, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                     [0.345421456, 0.524563927, ..., 0.144534654]],

                    [[0.345249859, 0.124939536, ..., 0.19435u375],
                     [0.945345345, 0.435394634, ..., 0.435345365]],

                    [[0.945345345, 0.435394634, ..., 0.435345365],
                     [0.0,         0.0,         ..., 0.0        ]]]  # padding data
    输入的padding_idx小于0，则自动转换为padding_idx = -1 + 128 = 127, 对于输入id为127的词，进行padding处理。
    
    Case 2:

    input是lod level 为1的LoDTensor, 且padding_idx = 0
        input.lod = [[2, 3]]
        input.data = [[1], [3], [2], [4], [0]]
        input.shape = [5, 1]
    若size = [128, 16]
    输出为LoDTensor:
        out.lod = [[2, 3]]
        out.shape = [5, 1, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452]],
                    [[0.345421456, 0.524563927, ..., 0.144534654]],
                    [[0.345249859, 0.124939536, ..., 0.19435u375]],
                    [[0.945345345, 0.435394634, ..., 0.435345365]],
                    [[0.0,         0.0,         ..., 0.0        ]]]  # padding data
    输入的padding_idx = 0，则对于输入id为0的词，进行padding处理。


参数：
    - **input** (Variable) - 存储id信息，数据类型必须为：int64。
    - **size** (tuple|list) - embedding矩阵的维度。必须包含两个元素，第一个元素为vocab_size(词表大小), 第二个为emb_size（embedding 层维度）。
    - **is_sparse** (bool) - 是否使用稀疏的更新方式，这个参数只会影响反向的梯度更新的性能，sparse更新速度更快。默认为False。
    - **is_distributed** (bool) - 是否使用分布式的方式存储embedding 矩阵，仅在多机分布式cpu训练中使用。默认为False。
    - **padding_idx** (int|long|None) - padding_idx需在区间[-vocab_size, vocab_size)，否则不生效，padding_idx<0时，padding_idx 会被改成 vocab_size + padding_idx，input中等于padding_index的id对应的embedding信息会被设置为0。如果为none，不作处理，默认为None。
    - **param_attr** (ParamAttr) - 可通过param_attr设置该层权重参数的初始化方式、学习率等，默认为None。
    - **dtype** (str) - 输出Tensor或LoDTensor的数据类型，数据类型必须为：float32，float64，默认为float32。

返回：input对应的embedding信息，数据类型和dtype定义的类型一致。

返回类型：Variable(Tensor|LoDTensor)

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
    emb = fluid.embedding(input=data, size=[128, 64])









