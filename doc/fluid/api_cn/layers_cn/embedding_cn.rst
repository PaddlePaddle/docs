.. _cn_api_fluid_layers_embedding:

embedding
-------------------------------

.. py:function:: paddle.fluid.layers.embedding(input, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

嵌入层(Embedding Layer)

该层用于查找由输入提供的id在查找表中的嵌入矩阵。查找的结果是input里每个ID对应的嵌入矩阵。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
    - **input** (Variable) - 包含IDs信息的int64的张量变量。输入IDs的值应该(0<= id < size[0])。
    - **size** (tuple|list) - 查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    - **is_sparse** (bool) - 代表是否用稀疏更新的标志
    - **is_distributed** (bool) - 是否从远程参数服务端运行查找表
    - **padding_idx** (int|long|None) - 只要查找发现（padding_idx）在ids中，就会输出全0填充的数据，如果为 ``None`` ，对输出结果无影响。如果（padding_idx<0），则（padding_idx<0）会自动转换为（size[0] + padding_idx）,默认为None。
    - **param_attr** (ParamAttr) - 该层参数
    - **dtype** (np.dtype|core.VarDesc.VarType|str) - 数据类型：float32,float_16,int等。

返回：张量，存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[128, 64])









