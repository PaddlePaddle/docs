.. _cn_api_fluid_layers_sequence_slice:

sequence_slice
-------------------------------

.. py:function:: paddle.static.nn.sequence_slice(input, offset, length, name=None)


实现 Sequence Slice (序列切片) 运算

从给定序列中截取子序列。截取依据为按照所给相对开始位置的 ``offset`` （偏移量）和子序列长度 ``length`` 来截取子序列。

.. note::
该 API 输入只能是带有 LoD 信息的 Tensor，如果您需要处理的是 Tensor 类型，请使用 :ref:`paddle.slice <cn_api_paddle_slice>` 。

.. code-block:: text

    输入：

    input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
    input.lod  = [[3, 2]],
    input.dims = [5, 2]

    offset.data = [[0], [1]]
    length.data = [[2], [1]]


    输出：

    out.data = [[a1, a2], [b1, b2], [e1, e2]],
    out.lod  = [[2, 1]],
    out.dims = [3, 2]


.. note::
    ``input``、``offset`` 以及 ``length`` 的第一维大小应相同。
    ``offset`` 从 0 开始。

参数
:::::::::
  - **input** (Tensor) – 输入变量，类型为 Tensor，承载着完整的序列。数据类型为 float32，float64，int32 或 int64。
  - **offset** (Tensor) – 指定每个序列切片的起始索引，数据类型为 int32 或 int64。
  - **length** (Tensor) – 指定每个子序列的长度，数据类型为 int32 或 int64。
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，序列切片运算结果。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_slice
