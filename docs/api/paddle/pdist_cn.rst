.. _cn_api_paddle_pdist:

pdist
-------------------------------

.. py:function:: paddle.pdist(x, p=2.0, name=None)

计算输入形状为 N x M 的 Tensor 中 N 个向量两两组合(pairwise)的 p 范数。


参数
::::::::::::

  - **x** (Tensor) - 输入的 Tensor，形状为 :math:`N \times M` 。
  - **p** (float， 可选) - 计算每个向量对之间的 p 范数距离的值。默认值为 :math:`2.0`。
  - **name** (str, 可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，数据类型与输入张量相同，输出的形状为 :math:`N \times (N-1) \div 2`。

代码示例
::::::::::::

COPY-FROM: paddle.pdist
