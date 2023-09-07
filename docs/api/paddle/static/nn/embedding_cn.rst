.. _cn_api_paddle_static_nn_embedding:

embedding
-------------------------------


.. py:function:: paddle.static.nn.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')




**嵌入层(Embedding Layer)**

该 OP 根据 input 中的 id 信息从 embedding 矩阵中查询对应 embedding 信息，并会根据输入的 size (vocab_size, emb_size)和 dtype 自动构造一个二维 embedding 矩阵。

输出的 Tensor 的 shape 是将输入 Tensor shape 的会在输出的 embedding 最后追加一维 emb_size。

.. note::
input 中的 id 必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


::

    Case 1:

    input 是 Tensor，且 padding_idx = -1
        input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
        input.shape = [3, 2, 1]
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

    输出为：
        out.lod = [[2, 3]]
        out.shape = [5, 16]
        out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                    [0.345421456, 0.524563927, ..., 0.144534654],
                    [0.345249859, 0.124939536, ..., 0.194353745],
                    [0.945345345, 0.435394634, ..., 0.435345365],
                    [0.0,         0.0,         ..., 0.0        ]]  # padding data
    输入的 padding_idx = 0，则对于输入 id 为 0 的词，进行 padding 处理。


参数
::::::::::::

    - **input** (Variable) - 存储 id 信息的 Tensor，数据类型必须为：int64，输入的 shape 最后一维须为 1。input 中的 id 必须满足 ``0 =< id < size[0]`` 。
    - **size** (tuple|list) - embedding 矩阵的维度。必须包含两个元素，第一个元素为 vocab_size(词表大小)，第二个为 emb_size（embedding 层维度）。
    - **is_sparse** (bool，可选) - 是否使用稀疏的更新方式，这个参数只会影响反向的梯度更新的性能，sparse 更新速度更快，推荐使用稀疏更新的方式。但某些 optimizer 不支持 sparse 更新，比如 :ref:`cn_api_paddle_optimizer_Adadelta` 、 :ref:`cn_api_paddle_optimizer_Adamax`，此时 is_sparse 必须为 False。默认为 False。
    - **is_distributed** (bool，可选) - 是否使用分布式的方式存储 embedding 矩阵，仅在多机分布式 cpu 训练中使用。默认为 False。
    - **padding_idx** (int|long|None，可选) - padding_idx 需在区间 ``[-vocab_size, vocab_size)``，否则不生效，``padding_idx < 0`` 时，padding_idx 会被改成``vocab_size + padding_idx``，input 中等于 padding_index 的 id 对应的 embedding 信息会被设置为 0，且这部分填充数据在训练时将不会被更新。如果为 None，不作处理，默认为 None。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。此外，可以通过 ``param_attr`` 参数加载用户自定义或预训练的词向量。只需将本地词向量转为 numpy 数据格式，且保证本地词向量的 shape 和 embedding 的 ``size`` 参数一致，然后使用 :ref:`cn_api_paddle_to_tensor` 进行初始化，即可实现加载自定义或预训练的词向量。
    - **dtype** (str，可选) - 输出 Tensor 的数据类型，数据类型必须为：float32 或 float64，默认为 float32。

返回
::::::::::::
Variable，input 映射后得到的 Embedding Tensor，数据类型和 dtype 定义的类型一致。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.embedding
