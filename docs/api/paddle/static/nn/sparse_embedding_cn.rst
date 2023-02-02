.. _cn_api_fluid_contrib_layers_sparse_embedding:

sparse_embedding
-------------------------------


.. py:function:: paddle.static.nn.sparse_embedding(input, size, padding_idx=None, is_test=False, entry=None, table_class="CommonSparseTable", param_attr=None, dtype='float32')


在飞桨参数服务器模式的大规模稀疏训练中作为 embedding lookup 层的算子，而不是使用 paddle.nn.functional.embedding。

根据 input 中的 id 信息从 embedding 矩阵中查询对应 embedding 信息，并会根据输入的 size (vocab_size, emb_size)和 dtype 自动构造一个二维 embedding 矩阵。

输出的 Tensor 的 shape 是将输入 Tensor shape 的会在输出的 embedding 最后追加一维 emb_size。

.. note::
input 中的 id 必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


::

    Case 1:

    input 是 Tensor，且 padding_idx = -1
        input.data = [[1, 3], [2, 4], [4, 127]]
        input.shape = [3, 2]
    若 size = [128, 16]
    输出为 Tensor:
        out.shape = [3, 2, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                     [0.345421456, 0.524563927, ..., 0.144534654]],

                    [[0.345249859, 0.124939536, ..., 0.194353745],
                     [0.945345345, 0.435394634, ..., 0.435345365]],

                    [[0.945345345, 0.435394634, ..., 0.435345365],
                     [0.0,         0.0,         ..., 0.0        ]]]  # padding data
    输入的 padding_idx 小于 0，则自动转换为 padding_idx = -1 + 128 = 127，对于输入 id 为 127 的词，进行 padding 处理。

    Case 2:

    input 是 lod level 为 1 的 Tensor，且 padding_idx = 0
        input.lod = [[2, 3]]
        input.data = [[1], [3], [2], [4], [0]]
        input.shape = [5, 1]

    若 size = [128, 16]

    输出为 Tensor:
        out.lod = [[2, 3]]
        out.shape = [5, 1, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452]],
                    [[0.345421456, 0.524563927, ..., 0.144534654]],
                    [[0.345249859, 0.124939536, ..., 0.194353745]],
                    [[0.945345345, 0.435394634, ..., 0.435345365]],
                    [[0.0,         0.0,         ..., 0.0        ]]]  # padding data
    输入的 padding_idx = 0，则对于输入 id 为 0 的词，进行 padding 处理。


参数
::::::::
    - **input** (Variable) - 存储 id 信息的 Tensor，数据类型必须为：int64，输入的 shape 最后一维须为 1。input 中的 id 必须满足 ``0 =< id < size[0]`` 。
    - **size** (tuple|list) - embedding 矩阵的维度(vocab_size，emb_size)。必须包含两个元素，第一个元素为 vocab_size(词表大小)，第二个为 emb_size（embedding 层维度）。大规模稀疏场景下，参数规模初始为 0，会随着训练的进行逐步扩展，因此如果 vocab_size 暂时无用，其值可以为任意整数，emb_size 则为词嵌入权重参数的维度配置。
    - **padding_idx** (int|long|None，可选) - padding_idx 需在区间 ``[-vocab_size, vocab_size)``，否则不生效，``padding_idx < 0`` 时，padding_idx 会被改成``vocab_size + padding_idx``，input 中等于 padding_index 的 id 对应的 embedding 信息会被设置为 0，且这部分填充数据在训练时将不会被更新。如果为 None，不作处理，默认为 None。
    - **is_test** (bool，可选) -  表示训练/预测模式。在预测模式(is_test=False)下，遇到不存在的特征，不会初始化及创建，直接以 0 填充后返回。默认值为 False。
    - **entry** (str，可选) - 准入策略配置，目前支持概率准入 ProbabilityEntry 和频次准入 CountFilterEntry。默认为 None。
    - **table_class** (str，可选) - 稀疏表的类型，其值可以为 CommonSparseTable 和 SSDSparseTable。默认为 CommonSparseTable。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。此外，可以通过 ``param_attr`` 参数加载用户自定义或预训练的词向量。只需将本地词向量转为 numpy 数据格式，且保证本地词向量的 shape 和 embedding 的 ``size`` 参数一致，然后使用 :ref:`cn_api_paddle_to_tensor` 进行初始化，即可实现加载自定义或预训练的词向量。
    - **dtype** (str) - 输出 Tensor 的数据类型，数据类型必须为：float32 或 float64，默认为 float32。

返回
::::::::
Variable，input 映射后得到的 Embedding Tensor，数据类型和 dtype 定义的类型一致。

代码示例
::::::::

COPY-FROM: paddle.static.nn.sparse_embedding
