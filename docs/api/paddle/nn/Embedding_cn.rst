.. _cn_api_nn_Embedding:

Embedding
-------------------------------

.. py:class:: paddle.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None)



嵌入层(Embedding Layer)

该接口用于构建 ``Embedding`` 的一个可调用对象，具体用法参照 ``代码示例`` 。其根据input中的id信息从embedding矩阵中查询对应embedding信息，并会根据输入的size (num_embeddings, embedding_dim)和weight_attr自动构造一个二维embedding矩阵。

输出的Tensor的shape是在输入Tensor shape的最后一维后面添加了embedding_dim的维度。

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
    - **num_embeddings** (int) - 嵌入字典的大小， input中的id必须满足 ``0 =< id < num_embeddings`` 。 。
    - **embedding_dim** (int) - 每个嵌入向量的维度。
    - **padding_idx** (int|long|None) - padding_idx的配置区间为 ``[-weight.shape[0], weight.shape[0]``，如果配置了padding_idx，那么在训练过程中遇到此id时会被用
    - **sparse** (bool) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
    - **weight_attr** (ParamAttr|None) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 :ref:`api_guide_ParamAttr` ，一般无需设置，默认值为None。
0填充。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回：input映射后得到的Embedding Tensor，数据类型和词嵌入的定义类型一致。

返回类型：Tensor

**代码示例**

.. code-block:: python

   import paddle
   import numpy as np

   x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
   y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)

   x = paddle.to_tensor(x_data, stop_gradient=False)
   y = paddle.to_tensor(y_data, stop_gradient=False)

   embedding = paddle.nn.Embedding(10, 3, sparse=True)

   w0=np.full(shape=(10, 3), fill_value=2).astype(np.float32)
   embedding.weight.set_value(w0)

   adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
   adam.clear_grad()

   # weight.shape = [10, 3]

   # x.data = [[3],[4],[5]]
   # x.shape = [3, 1]

   # out.data = [[2,2,2], [2,2,2], [2,2,2]]
   # out.shape = [3, 1, 3]
   out=embedding(x)
   out.backward()
   adam.step()


