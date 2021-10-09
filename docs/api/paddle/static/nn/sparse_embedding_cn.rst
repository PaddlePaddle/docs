.. _cn_api_fluid_input_embedding:

sparse_embedding
-------------------------------


.. py:function:: paddle.static.nn.sparse_embedding(input, size, padding_idx=None, is_test=False, entry=None, table_class="CommonSparseTable", param_attr=None, dtype='float32')


该OP在飞桨参数服务器模式的大规模稀疏训练中作为embedding lookup层的算子，而不是使用paddle.nn.functional.embedding。

该OP根据input中的id信息从embedding矩阵中查询对应embedding信息，并会根据输入的size (vocab_size, emb_size)和dtype自动构造一个二维embedding矩阵。

输出的Tensor的shape是将输入Tensor shape的会在输出的embedding最后追加一维emb_size。

注：input中的id必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


::

    Case sparse_embedding_cn.rst1:

    input是Tensor, 且padding_idx = -1
        input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
        input.shape = [3, 2, 1]
    若size = [128, 16]
    输出为Tensor:
        out.shape = [3, 2, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                     [0.345421456, 0.524563927, ..., 0.144534654]],

                    [[0.345249859, 0.124939536, ..., 0.194353745],
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

    输出为:
        out.lod = [[2, 3]]
        out.shape = [5, 16]
        out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                    [0.345421456, 0.524563927, ..., 0.144534654],
                    [0.345249859, 0.124939536, ..., 0.194353745],
                    [0.945345345, 0.435394634, ..., 0.435345365],
                    [0.0,         0.0,         ..., 0.0        ]]  # padding data
    输入的padding_idx = 0，则对于输入id为0的词，进行padding处理。


参数：
    - **input** (Variable) - 存储id信息的Tensor，数据类型必须为：int64，输入的shape最后一维须为1。input中的id必须满足 ``0 =< id < size[0]`` 。
    - **size** (tuple|list) - embedding矩阵的维度(vocab_size，emb_size)。必须包含两个元素，第一个元素为vocab_size(词表大小), 第二个为emb_size（embedding层维度）。大规模稀疏场景下，参数规模初始为0，会随着训练的进行逐步扩展，因此如果vocab_size暂时无用，其值可以为任意整数，emb_size则为词嵌入权重参数的维度配置。
    - **padding_idx** (int|long|None，可选) - padding_idx需在区间 ``[-vocab_size, vocab_size)`` ，否则不生效，``padding_idx < 0`` 时，padding_idx会被改成``vocab_size + padding_idx``，input中等于padding_index的id对应的embedding信息会被设置为0，且这部分填充数据在训练时将不会被更新。如果为None，不作处理，默认为None。
    - **is_test** (bool，可选) -  表示训练/预测模式。在预测模式(is_test=False)下，遇到不存在的特征，不会初始化及创建，直接以0填充后返回。默认值为False。
    - **entry** (str，可选) - 准入策略配置，目前支持概率准入ProbabilityEntry和频次准入CountFilterEntry。默认为None。
    - **table_class** (str，可选) - 稀疏表的类型，其值可以为CommonSparseTable和SSDSparseTable。默认为CommonSparseTable。 
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。此外，可以通过 ``param_attr`` 参数加载用户自定义或预训练的词向量。只需将本地词向量转为numpy数据格式，且保证本地词向量的shape和embedding的 ``size`` 参数一致，然后使用 :ref:`cn_api_paddle_to_tensor` 进行初始化，即可实现加载自定义或预训练的词向量。
    - **dtype** (str|core.VarDesc.VarType) - 输出Tensor的数据类型，数据类型必须为：float32 或float64，默认为float32。

返回：input映射后得到的Embedding Tensor或LoDTensor，数据类型和dtype定义的类型一致。

返回类型：Variable

**代码示例**:

.. code-block:: python

    import paddle
    paddle.enable_static()
    sparse_feature_dim = 1024
    embedding_size = 64
    
    # 训练过程中，出现超过10次及以上的特征才会参与训练
    entry = paddle.distributed.CountFilterEntry(10)

    input = paddle.static.data(name='ins', shape=[1], dtype='int64')

    emb = paddle.static.nn.sparse_embedding((
       input=input,
       size=[sparse_feature_dim, embedding_size],
       is_test=False,
       entry=entry,
       param_attr=paddle.ParamAttr(name="SparseFeatFactors",
       initializer=paddle.nn.initializer.Uniform()))


