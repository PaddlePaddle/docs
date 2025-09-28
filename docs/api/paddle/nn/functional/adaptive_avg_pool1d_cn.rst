.. _cn_api_paddle_nn_functional_adaptive_avg_pool1d:

adaptive_avg_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_avg_pool1d(x, output_size, name=None)

根据 ``output_size`` 对 Tensor ``x`` 计算 1D 自适应平均池化。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_paddle_nn_AdaptiveAvgPool1D`。


参数
:::::::::
    - **x** (Tensor) - 自适应平均池化的输入，它是形状为 :math:`[N,C,L]` 的 3-D Tensor，其中 :math:`N` 是批大小，:math:`C` 是通道数而 :math:`L` 是输入特征的长度，其数据类型为 float32 或 float64。
    - **output_size** (int) - 输出特征的长度，数据类型为 int。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，计算 1D 自适应平均池化的结果，数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.adaptive_avg_pool1d
