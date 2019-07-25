.. _cn_api_fluid_dygraph_Embedding:

Embedding
-------------------------------

.. py:class:: paddle.fluid.dygraph.Embedding(name_scope, size, is_sparse=False, is_distributed=False, padding_idx=None, param_attr=None, dtype='float32')

Embedding层

该层用于在查找表中查找 ``input`` 中的ID对应的embeddings。查找的结果是input里每个ID对应的embedding。
所有的输入变量都作为局部变量传入LayerHelper构造器

参数：
    - **name_scope** (str)-该类的名称。
    - **size** (tuple|list)-查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    - **is_sparse** (bool)-代表是否用稀疏更新的标志。
    - **is_distributed** (bool)-是否从远程参数服务端运行查找表。
    - **padding_idx** (int|long|None)-如果为 ``None`` ，对查找结果无影响。如果 ``padding_idx`` 不为空，表示一旦查找表中找到input中对应的 ``padding_idx``，则用0填充输出结果。如果 ``padding_idx`` <0 ,则在查找表中使用的 ``padding_idx`` 值为 :math:`size[0]+dim` 。默认：None。
    - **param_attr** (ParamAttr)-该层参数。默认为None。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-数据类型：float32,float_16,int等。默认:‘float32’

返回：张量，存储已有输入的嵌入矩阵。

返回类型：变量(Variable)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.dygraph.base as base
    import numpy as np

    inp_word = np.array([[[1]]]).astype('int64')
    dict_size = 20
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding(
            name_scope='embedding',
            size=[dict_size, 32],
            param_attr='emb.w',
            is_sparse=False)
        static_rlt3 = emb(base.to_variable(inp_word))





