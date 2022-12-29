.. _cn_api_fluid_layers_sequence_expand:

sequence_expand
-------------------------------


.. py:function:: paddle.static.nn.sequence_expand(x, y, ref_level=-1, name=None)



序列扩张层（Sequence Expand Layer)，根据输入 ``y`` 的第 ``ref_level`` 层 lod 对输入 ``x`` 进行扩展。``x`` 的 lod level 最多为 1，若 ``x`` 的 lod level 为 1，则 ``x`` 的 lod 大小必须与 ``y`` 的第 ``ref_level`` 层 lod 大小相等；若 ``x`` 的 lod level 为 0，则 ``x`` 的第一维大小必须与 ``y`` 第 ``ref_level`` 层大小相等。``x`` 的秩最少为 2，当 ``x`` 的秩大于 2 时，将被当作是一个二维 Tensor 处理。

.. note::
该 API 的输入 ``x`` 可以是 Tensor 或 LodTensor， ``y`` 只能是 LodTensor。

范例解释如下：

::

    例 1：
    假设两个长度为 2 的序列[a][b]和[c][d]，欲将其扩展为 4 个长度为 2 的序列[a][b]、[a][b]、[c][d]、[c][d]。
    序列[a][b]扩展 2 次，[c][d]扩展 2 次，扩展所需依据的 lod 为[2, 2]，则：
    给定输入一维 Tensor x
      x.lod  = [[2,        2]]    #表示两个序列的长度为 2，为了便于理解这里用基于长度 lod 表示
      x.data = [[a], [b], [c], [d]]
      x.dims = [4, 1]
    和输入 y
      y.lod = [[2,    2],     #第 0 层 lod，指定按该层扩展，表示分别扩展 2 次，为了便于理解这里用基于长度 lod 表示
               [3, 3, 1, 1]]  #第 1 层 lod，注意，因为指定 ref_level 为 0，所以这一层与运算无关
    指定 ref_level = 0，依据 y 的第 0 层 lod 进行扩展，

    经过 sequence_expand，输出为 1 级 Tensor out
      out.lod =  [[0,        2,        4,        6,      8]]  #基于偏移的 lod，等价于基于长度的[[2, 2, 2, 2]]
      out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
      out.dims = [8, 1]

::

    例 2：
    假设有 3 个长度维 1 的序列[a]、[b]、[c]，现在要将其扩展为长度是 2、0、3 的序列[a][a]、[c][c][c]。
    显然，扩展后的序列 lod 为[2, 0, 3]，则：
    给定输入一维 Tensor x
      x.data = [[a], [b], [c]]
      x.dims = [3, 1]
    和输入 y
      y.lod = [[2, 0, 3]]
    默认 ref_level = -1

    经过 sequence_expand，输出为 1 级 Tensor out
      out.data = [[a], [a], [c], [c], [c]]
      out.dims = [5, 1]

参数
:::::::::

    - **x** (Variable) - 输入变量，维度为 :math:`[M, K]` ，lod level 至多 1 的二维 Tensor。数据类型支持 int32，int64，float32 或 float64。
    - **y** (Variable) - 输入变量，lod level 至少为 1 的 Tensor。数据类型不限。
    - **ref_level** (int，可选) - 扩展 ``x`` 所依据的 ``y`` 的 lod 层。默认值-1，表示 lod 的最后一层。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
扩展变量，维度为 :math:`[N, K]` 的 Tensor，N 由输入 ``x`` 和 ``y`` 的 lod 共同决定。数据类型与输入 ``x`` 一致。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_expand
