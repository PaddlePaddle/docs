.. _cn_api_fluid_layers_multiplex:

multiplex
-------------------------------

.. py:function:: paddle.multiplex(inputs, index, name=None)



根据给定的index参数，从每个输入Tensor中选择特定行构造输出Tensor。

设输入包含 :math:`m` 个Tensor，其中 :math:`I_{i}` 代表第i个输入Tensor，:math:`i` 处于区间 :math:`[0,m)`。

设输出为 :math:`O`，其中 :math:`O[i]` 为输出的第i行，则输出满足：:math:`O[i] = I_{index[i]}[i]`

示例：


COPY-FROM: paddle.multiplex

参数
::::::::::::

  - **inputs** （list） - 为输入Tensor列表，列表元素为数据类型为float32、float64、int32、int64的多维Tensor。所有输入Tensor的shape应相同，秩必须至少为2。
  - **index** （Tensor）- 用来选择输入Tensor中的某些行构建输出Tensor的索引，为数据类型为int32或int64、shape为[M, 1]的2-D Tensor，其中M为输入Tensor个数。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，进行Multiplex运算后的输出Tensor。

代码示例
::::::::::::


COPY-FROM: paddle.multiplex:code-example1

