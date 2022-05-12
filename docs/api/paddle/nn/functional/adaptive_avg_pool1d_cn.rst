.. _cn_api_nn_functional_adaptive_avg_pool1d:


adaptive_avg_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_avg_pool1d(x, output_size, name=None)

该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算1D的自适应平均池化。输入和输出都是3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数, `L` 是输入特征的长度.

.. note::
   详细请参考对应的 `Class` 请参考: :ref:`cn_api_nn_AdaptiveAvgPool1D` 。


参数
:::::::::
    - **x** (Tensor): 当前算子的输入, 其是一个形状为 `[N, C, L]` 的3-D Tensor。其中 `N` 是batch size, `C` 是通道数, `L` 是输入特征的长度。 其数据类型为float32或者float64。
    - **output_size** (int): 算子输出特征图的长度，其数据类型为int。
    - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输入 `x` 经过自适应池化计算得到的目标3-D Tensor，其数据类型与输入相同。


代码示例
:::::::::
COPY-FROM: paddle.nn.functional.adaptive_avg_pool1d
