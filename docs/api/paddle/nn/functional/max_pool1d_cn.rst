.. _cn_api_paddle_nn_functional_max_pool1d:


max_pool1d
-------------------------------

.. py:function:: paddle.nn.functional.max_pool1d(x, kernel_size, stride=None, padding=0, return_mask=False, ceil_mode=False, name=None)

根据输入 `x` , `kernel_size` 等参数对一个输入 Tensor 计算 1D 的最大值池化。输入和输出都是 3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。

.. note::
   详细请参考对应的 `Class` 请参考：:ref:`cn_api_paddle_nn_MaxPool1D` 。


参数
:::::::::
    - **x** (Tensor) - 当前算子的输入，其是一个形状为 `[N, C, L]` 的 3-D Tensor。其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。其数据类型为 float32 或者 float64。
    - **kernel_size** (int|list|tuple) - 池化核的尺寸大小。如果 kernel_size 为 list 或 tuple 类型，其必须包含一个整数。
    - **stride** (int|list|tuple) - 池化操作步长。如果 stride 为 list 或 tuple 类型，其必须包含一个整数。
    - **padding** (string|int|list|tuple) - 池化补零的方式。如果 padding 是一个字符串，则必须为 `SAME` 或者 `VALID`。如果是 turple 或者 list 类型，则应是 `[pad_left, pad_right]` 形式。如果 padding 是一个非 0 值，那么表示会在输入的两端都 padding 上同样长度的 0。
    - **return_mask** (bool) - 是否返回最大值的索引，默认为 False。
    - **ceil_mode** (bool) - 是否用 ceil 函数计算输出的 height 和 width，如果设置为 False，则使用 floor 函数来计算，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。




返回
:::::::::
``Tensor``，输入 `x` 经过最大值池化计算得到的目标 3-D Tensor，其数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.max_pool1d
