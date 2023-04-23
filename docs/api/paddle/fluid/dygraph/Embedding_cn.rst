.. _cn_api_fluid_dygraph_Embedding:

Embedding
-------------------------------

.. py:class:: paddle.fluid.dygraph.Embedding(size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')




嵌入层(Embedding Layer)

该接口用于构建 ``Embedding`` 的一个可调用对象，具体用法参照 ``代码示例``。其根据 input 中的 id 信息从 embedding 矩阵中查询对应 embedding 信息，并会根据输入的 size (vocab_size, emb_size)和 dtype 自动构造一个二维 embedding 矩阵。

输出的 Tensor 的 shape 是在输入 Tensor shape 的最后一维后面添加了 emb_size 的维度。

注：input 中的 id 必须满足 ``0 =< id < size[0]``，否则程序会抛异常退出。


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

    input 是 lod level 为 1 的 LoDTensor，且 padding_idx = 0
        input.lod = [[2, 3]]
        input.data = [[1], [3], [2], [4], [0]]
        input.shape = [5, 1]
    若 size = [128, 16]
    输出为 LoDTensor:
        out.lod = [[2, 3]]
        out.shape = [5, 1, 16]
        out.data = [[[0.129435295, 0.244512452, ..., 0.436322452]],
                    [[0.345421456, 0.524563927, ..., 0.144534654]],
                    [[0.345249859, 0.124939536, ..., 0.194353745]],
                    [[0.945345345, 0.435394634, ..., 0.435345365]],
                    [[0.0,         0.0,         ..., 0.0        ]]]  # padding data
    输入的 padding_idx = 0，则对于输入 id 为 0 的词，进行 padding 处理。

参数
::::::::::::

    - **size** (tuple|list) - embedding 矩阵的维度。必须包含两个元素，第一个元素为 vocab_size(词表大小)，第二个为 emb_size（embedding 层维度）。
    - **is_sparse** (bool) - 是否使用稀疏的更新方式，这个参数只会影响反向的梯度更新的性能，sparse 更新速度更快，推荐使用稀疏更新的方式。但某些 optimizer 不支持 sparse 更新，比如 :ref:`cn_api_fluid_optimizer_AdadeltaOptimizer` 、 :ref:`cn_api_fluid_optimizer_AdamaxOptimizer` 、 :ref:`cn_api_fluid_optimizer_DecayedAdagradOptimizer` 、 :ref:`cn_api_fluid_optimizer_FtrlOptimizer` 、 :ref:`cn_api_fluid_optimizer_LambOptimizer` 、:ref:`cn_api_fluid_optimizer_LarsMomentumOptimizer`，此时 is_sparse 必须为 False。默认为 False。
    - **is_distributed** (bool) - 是否使用分布式的方式存储 embedding 矩阵，仅在多机分布式 cpu 训练中使用。默认为 False。
    - **padding_idx** (int|long|None) - padding_idx 需在区间[-vocab_size, vocab_size)，否则不生效，padding_idx<0 时，padding_idx 会被改成 vocab_size + padding_idx，input 中等于 padding_index 的 id 对应的 embedding 信息会被设置为 0，且这部分填充数据在训练时将不会被更新。如果为 None，不作处理，默认为 None。
    - **param_attr** (ParamAttr) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。此外，可以通过 ``param_attr`` 参数加载用户自定义或预训练的词向量。只需将本地词向量转为 numpy 数据格式，且保证本地词向量的 shape 和 embedding 的 ``size`` 参数一致，然后使用 :ref:`cn_api_fluid_initializer_NumpyArrayInitializer` 进行初始化，即可实现加载自定义或预训练的词向量。详细使用方法见代码示例 2。
    - **dtype** (str|core.VarDesc.VarType) - 输出 Tensor 的数据类型，数据类型必须为：float32 或 float64，默认为 float32。

返回
::::::::::::
input 映射后得到的 Embedding Tensor，数据类型和 dtype 定义的类型一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy as np

    # 示例 1
    inp_word = np.array([[2, 3, 5], [4, 2, 1]]).astype('int64')
    inp_word.shape  # [2, 3]
    dict_size = 20
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding(
            size=[dict_size, 32],
            param_attr='emb.w',
            is_sparse=False)
        static_rlt3 = emb(base.to_variable(inp_word))
        static_rlt3.shape  # [2, 3, 32]

    # 示例 2：加载用户自定义或预训练的词向量
    weight_data = np.random.random(size=(128, 100))  # numpy 格式的词向量数据
    w_param_attrs = fluid.ParamAttr(
        name="emb_weight",
        learning_rate=0.5,
        initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
        trainable=True)
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding(
            size=[128, 100],
            param_attr= w_param_attrs,
            is_sparse=False)
        static_rlt3 = emb(base.to_variable(inp_word))
