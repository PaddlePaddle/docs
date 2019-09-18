.. _cn_api_fluid_layers_embedding:

embedding
-------------------------------

.. py:function:: paddle.fluid.layers.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

嵌入层(Embedding Layer)

该层包含一个指定大小的词嵌入张量矩阵，会根据 ``input`` 中提供的 ``id`` 作为索引，在词嵌入矩阵中查询每个 ``id`` 对应的张量作为输出。

参数：
    - **input** (Variable) - 包含 ``id`` 索引信息的Tensor或LoDTensor，取值必须为int64类型，且输入的 ``id`` 值范围须满足 :math:`0 <= id < size[0]`；输入的shape最后一维须为1。
    - **size** (tuple|list) - 用于指定词嵌入张量矩阵的维度 :math`(vocab_size, emb_dim)` ，第一个参数表示词嵌入矩阵字典的大小，第二个表示词嵌入向量的维度。
    - **is_sparse** (bool) - 若设置为 ``True`` , 则采用稀疏更新的方式进行参数更新，默认为 ``False`` 。
    - **is_distributed** (bool) - 若设置为 `True` , 则从远程参数服务端获取词嵌入张量矩阵，默认为 ``False`` 。
    - **padding_idx** (int|long|None) - 若给定 ``padding\_idx`` 参数，只要输入的 ``id`` 取值为 ``padding\_idx`` , 则会输出全0填充的词向量；若 :math`padding\_idx < 0` ，则 ``padding\_idx`` 会自动转换为 :math`（size[0] + padding\_idx）` ，默认为 ``None`` 。
    - **param_attr** (ParamAttr) - 可通过 ``param_attr`` 设置该层权重参数的初始化方式、学习率等，默认为 ``None`` 。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) - 输出张量的数据类型，如float32，float_16，int等类型，默认为float32。

返回：存储 ``input`` 映射后的词嵌入矩阵。

返回类型：Variable(Tensor|LoDTensor)

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[128, 64])









