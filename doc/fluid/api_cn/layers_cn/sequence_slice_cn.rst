.. _cn_api_fluid_layers_sequence_slice:

sequence_slice
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_slice(input, offset, length, name=None)

**实现Sequence Slice(序列切片)运算**

该层从给定序列中截取子序列。截取依据为所给的开始 ``offset`` （偏移量） 和子序列长 ``length`` 。

仅支持序列数据，LoD level（LoD层次为1）
::
    输入变量：

        input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
        input.lod = [[3, 2]],
        input.dims = (5, 2),

    以及 offset.data = [[0], [1]] and length.data = [[2], [1]],

    则输出变量为：

        out.data = [[a1, a2], [b1, b2], [e1, e2]],
        out.lod = [[2, 1]],
        out.dims = (3, 2).

.. note::
   ``input`` ， ``offset`` ， ``length`` 的第一维大小应相同。
   ``offset`` 从0开始。

参数:
  - **input** (Variable) – 输入变量 ，承载着完整的序列
  - **offset** (Variable) – 对每个序列切片的起始索引
  - **length** (Variable) – 每个子序列的长度
  - **name** (str|None) – 该层的命名，可选项。 如果None, 则自动命名该层

返回：输出目标子序列

返回类型：Variable

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  import numpy as np
  seqs = fluid.layers.data(name='x', shape=[10, 5],
       dtype='float32', lod_level=1)
  offset = fluid.layers.assign(input=np.array([[0, 1]]).astype("int32"))
  length = fluid.layers.assign(input=np.array([[2, 1]]).astype("int32"))
  subseqs = fluid.layers.sequence_slice(input=seqs, offset=offset,
                length=length)










