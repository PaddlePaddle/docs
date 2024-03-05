.. _cn_api_paddle_nn_Embedding:

Embedding
-------------------------------

.. py:class:: paddle.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None)

嵌入层(Embedding Layer)，用于构建 ``Embedding`` 的一个可调用对象，具体用法参照 ``代码示例``。其根据 ``x`` 中的 id 信息从 embedding 矩阵中查询对应 embedding 信息，并会根据输入的 size (num_embeddings, embedding_dim)和 weight_attr 自动构造一个二维 embedding 矩阵。

输出的 Tensor 的 shape 是在输入 Tensor shape 的最后一维后面添加了 embedding_dim 的维度。

.. note::
   input 中的 id 必须满足 ``0 <= id < size[0]``，否则程序会抛异常退出。

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

    - **num_embeddings** (int) - 嵌入字典的大小，input 中的 id 必须满足 ``0 <= id < num_embeddings`` 。
    - **embedding_dim** (int) - 每个嵌入向量的维度。
    - **padding_idx** (int|long|None，可选) - padding_idx 的配置区间为 ``[-weight.shape[0], weight.shape[0]]``，如果配置了 padding_idx，那么在训练过程中遇到此 id 时，其参数及对应的梯度将会以 0 进行填充。
    - **sparse** (bool，可选) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
    - **weight_attr** (ParamAttr|None，可选) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 :ref:`api_guide_ParamAttr`，一般无需设置，默认值为 None。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


属性
:::::::::

weight
'''''''''

本层的可学习参数，类型为 ``Parameter`` 。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.Embedding
