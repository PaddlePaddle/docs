.. _cn_api_nn_functional_embedding:

embedding
-------------------------------


.. py:function:: paddle.nn.functional.embedding(x, weight, padding_idx=None, sparse=False, name=None)



嵌入层(Embedding Layer)

该OP根据input中的id信息从embedding矩阵中查询对应embedding信息，并会根据输入的size (vocab_size, emb_size)和dtype自动构造一个二维embedding矩阵。

输出的Tensor的shape是将输入Tensor shape后追加一维emb_size。

注：input中的id必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


.. code-block:: text

            x是Tensor， 且padding_idx = -1.
                padding_idx = -1
                x.data = [[1, 3], [2, 4], [4, 127]]
                x.shape = [3, 2]
                weight.shape = [128, 16]
            输出是Tensor:
                out.shape = [3, 2, 16]
                out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654]],
                            [[0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365]],
                            [[0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]]  # padding data

            输入的padding_idx小于0，则自动转换为padding_idx = -1 + 128 = 127, 对于输入id为127的词，进行padding处理。


参数：

    - **input** (Tensor) - 存储id信息的Tensor，数据类型必须为：int32/int64。input中的id必须满足 ``0 =< id < size[0]`` 。
    - **weight** (Tensor) - 存储词嵌入权重参数的Tensor，形状为(num_embeddings, embedding_dim)。
    - **sparse** (bool) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
    - **padding_idx** (int|long|None) - padding_idx的配置区间为 ``[-weight.shape[0], weight.shape[0]``，如果配置了padding_idx，那么在训练过程中遇到此id时会被用0填充。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回：input映射后得到的Embedding Tensor，数据类型和权重定义的类型一致。

返回类型：Tensor

**代码示例**:

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn

    x0 = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
    w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)

    # x.data = [[3], [4], [5]]
    # x.shape = [3, 1]
    x = paddle.to_tensor(x0, stop_gradient=False)

    # w.data = [[2. 2. 2.] ... [2. 2. 2.]]
    # w.shape = [10, 3]
    w = paddle.to_tensor(w0, stop_gradient=False)

    # emb.data = [[[2., 2., 2.]], [[2., 2., 2.]], [[2., 2., 2.]]]
    # emb.shape = [3, 1, 3]

    emb = nn.functional.embedding(
            x=x, weight=w, sparse=True, name="embedding")


