.. _cn_api_fluid_layers_sequence_slice:

sequence_slice
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_slice(input, offset, length, name=None)




**实现Sequence Slice(序列切片)运算**

**该OP输入只能是LoDTensor，如果您需要处理的是Tensor类型，请使用 :ref:`cn_api_fluid_layers_slice` 。**
该层从给定序列中截取子序列。截取依据为所给的开始 ``offset`` （偏移量） 和子序列长 ``length`` 。

::
    输入变量：
        (1) input (LoDTensor):
                input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
                input.lod = [[3, 2]],
                input.dims = (5, 2),

        (2) offset (Variable):
                offset.data = [[0], [1]]
        (3) length (Variable):
                length.data = [[2], [1]]
        (4) name (str|None)

    输出变量为LoDTensor：

        out.data = [[a1, a2], [b1, b2], [e1, e2]],
        out.lod = [[2, 1]],
        out.dims = (3, 2).

.。注意：:
   ``input`` ， ``offset`` ， ``length`` 的第一维大小应相同。
   ``offset`` 从0开始。

参数
::::::::::::

  - **input** (Variable) – 输入变量，类型为LoDTensor，承载着完整的序列。数据类型为float32，float64，int32或int64。
  - **offset** (Variable) – 指定每个序列切片的起始索引，数据类型为int32或int64。
  - **length** (Variable) – 指定每个子序列的长度，数据类型为int32或int64。
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
::::::::::::
Variable(LoDTensor) 序列切片运算结果

返回类型
::::::::::::
变量(Variable)，数据类型与 ``input`` 一致

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_slice