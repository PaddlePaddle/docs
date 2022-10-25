.. _cn_api_fluid_layers_sequence_expand:

sequence_expand
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_expand(x, y, ref_level=-1, name=None)




序列扩张层（Sequence Expand Layer)，根据输入 ``y`` 的第 ``ref_level`` 层lod对输入 ``x`` 进行扩展。``x`` 的lod level最多为1，若 ``x`` 的lod level为1，则 ``x`` 的lod大小必须与 ``y`` 的第 ``ref_level`` 层lod大小相等；若 ``x`` 的lod level为0，则 ``x`` 的第一维大小必须与 ``y`` 第 ``ref_level`` 层大小相等。``x`` 的秩最少为2，当 ``x`` 的秩大于2时，将被当作是一个二维张量处理。

注意，该OP的输入 ``x`` 可以是Tensor或LodTensor， ``y`` 只能是LodTensor。

范例解释如下：

::

    例1：
    假设两个长度为2的序列[a][b]和[c][d]，欲将其扩展为4个长度为2的序列[a][b]、[a][b]、[c][d]、[c][d]。
    序列[a][b]扩展2次，[c][d]扩展2次，扩展所需依据的lod为[2, 2]，则：
    给定输入一维LoDTensor x
      x.lod  = [[2,        2]]    #表示两个序列的长度为2，为了便于理解这里用基于长度lod表示
      x.data = [[a], [b], [c], [d]]
      x.dims = [4, 1]
    和输入 y
      y.lod = [[2,    2],     #第0层lod，指定按该层扩展，表示分别扩展2次，为了便于理解这里用基于长度lod表示
               [3, 3, 1, 1]]  #第1层lod，注意，因为指定ref_level为0，所以这一层与运算无关
    指定 ref_level = 0，依据y的第0层lod进行扩展，

    经过sequence_expand，输出为1级LoDTensor out
      out.lod =  [[0,        2,        4,        6,      8]]  #基于偏移的lod，等价于基于长度的[[2, 2, 2, 2]]
      out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
      out.dims = [8, 1]

::

    例2：
    假设有3个长度维1的序列[a]、[b]、[c]，现在要将其扩展为长度是2、0、3的序列[a][a]、[c][c][c]。
    显然，扩展后的序列lod为[2, 0, 3]，则：
    给定输入一维LoDTensor x
      x.data = [[a], [b], [c]]
      x.dims = [3, 1]
    和输入 y
      y.lod = [[2, 0, 3]]
    默认 ref_level = -1

    经过sequence_expand，输出为1级LoDTensor out
      out.data = [[a], [a], [c], [c], [c]]
      out.dims = [5, 1]

参数
::::::::::::

    - **x** (Variable) - 输入变量，维度为 :math:`[M, K]` ，lod level至多1的二维Tensor或LoDTensor。数据类型支持int32，int64，float32或float64。
    - **y** (Variable) - 输入变量，lod level至少为1的LoDTensor。数据类型不限。
    - **ref_level** (int，可选) - 扩展 ``x`` 所依据的 ``y`` 的lod层。默认值-1，表示lod的最后一层。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
扩展变量，维度为 :math:`[N, K]` 的LoDTensor，N由输入 ``x`` 和 ``y`` 的lod共同决定。数据类型与输入 ``x`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_expand