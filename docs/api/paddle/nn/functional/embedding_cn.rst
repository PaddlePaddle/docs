.. _cn_api_paddle_nn_functional_embedding:

embedding
-------------------------------

.. py:function:: paddle.nn.functional.embedding(x, weight, padding_idx=None, sparse=False, name=None)



嵌入层(Embedding Layer)，根据 input 中的 id 信息从 embedding 矩阵中查询对应 embedding 信息，并会根据输入的 size (vocab_size, emb_size)和 dtype 自动构造一个二维 embedding 矩阵。

输出的 Tensor 的 shape 是将输入 Tensor shape 后追加一维 emb_size。

.. note::

   input 中的 id 必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


.. code-block:: text

            x 是 Tensor，且 padding_idx = -1。
                padding_idx = -1
                x.data = [[1, 3], [2, 4], [4, 127]]
                x.shape = [3, 2]
                weight.shape = [128, 16]
            输出是 Tensor:
                out.shape = [3, 2, 16]
                out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654]],
                            [[0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365]],
                            [[0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]]  # padding data

            输入的 padding_idx 小于 0，则自动转换为 padding_idx = -1 + 128 = 127，对于输入 id 为 127 的词，进行 padding 处理。


参数
::::::::::::


    - **input** (Tensor) - 存储 id 信息的 Tensor，数据类型必须为：int32/int64。input 中的 id 必须满足 ``0 =< id < size[0]`` 。
    - **weight** (Tensor) - 存储词嵌入权重参数的 Tensor，形状为(num_embeddings, embedding_dim)。
    - **sparse** (bool，可选) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新（即设置为 True）能够获得更快的训练速度及更小的内存/显存占用。但是一些优化器不支持稀疏更新，例如 :ref:`cn_api_paddle_optimizer_Adadelta` ， :ref:`cn_api_paddle_optimizer_Adamax` ， :ref:`cn_api_paddle_optimizer_Lamb` 。在这些情况下，稀疏必须为 False。默认值：False。
    - **padding_idx** (int|long|None，可选) - padding_idx 的配置区间为 ``[-weight.shape[0], weight.shape[0]``，如果配置了 padding_idx，那么在训练过程中遇到此 id 时，其参数及对应的梯度将会以 0 进行填充。如果 padding_idx < 0 ，则 padding_idx 将自动转换到 ``weight.shape[0] + padding_idx`` 。如果设置为 "None"，则不会对输出产生影响。默认值：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
Tensor, input 映射后得到的 Embedding Tensor，数据类型和权重定义的类型一致。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.embedding
