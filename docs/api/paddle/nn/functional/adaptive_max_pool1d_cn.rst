.. _cn_api_paddle_nn_functional_adaptive_max_pool1d:


adaptive_max_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_max_pool1d(x, output_size, return_mask=False, name=None)

根据输入 `x` , `output_size` 等参数对一个输入 Tensor 计算 1D 的自适应最大值池化。输入和输出都是 3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_paddle_nn_AdaptiveMaxPool1D` 。


参数
:::::::::
    - **x** (Tensor)：当前算子的输入，其是一个形状为 `[N, C, L]` 的 3-D Tensor。其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。其数据类型为 float32 或者 float64。
    - **output_size** (int|list|tuple)：算子输出特征图的长度，其数据类型为 int 或 list，tuple。
    - **return_mask** (bool)：如果设置为 True，则会与输出一起返回最大值的索引，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输入 `x` 经过自适应池化计算得到的目标 3-D Tensor，其数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.adaptive_max_pool1d
