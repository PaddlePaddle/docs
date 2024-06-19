.. _cn_api_paddle_nn_LPPool1D:

LPPool1D
-------------------------------

.. py:function:: paddle.nn.LPPool1D(norm_type, kernel_size, stride=None, padding=0, ceil_mode=False, data_format="NCL", name=None)

根据输入 `x` , `kernel_size` 等参数对一个输入 Tensor 计算 1D 的幂平均池化。输入和输出都是 3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。

假设输入形状是(N, C, L)，输出形状是 (N, C, L_{out})，卷积核尺寸是 k, 1d 平均池化计算公式如下：

.. math::

    Output(N_i, C_i, l) = sum(Input[N_i, C_i, stride \times l:stride \times l+k]^{norm\_type})^{1/norm\_type}

参数
:::::::::
    - **norm_type** (int|float)：幂平均池化的指数，不可以为 0。
    - **kernel_size** (int|list|tuple) - 池化核的尺寸大小。如果 kernel_size 为 list 或 tuple 类型，其必须包含一个整数，最终池化核的大小为该数值。
    - **stride** (int|list|tuple，可选) - 池化操作步长。如果 stride 为 list 或 tuple 类型，其必须包含一个整数，最终池化操作的步长为该数值。默认值为 None，这时会使用 kernel_size 作为 stride。
    - **padding** (string|int|list|tuple) 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 2 个整数值：[pad_height, pad_width]；(2)包含 4 个整数值：[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 4 个二元组：当 data_format 为"NCHW"时为 [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NHWC"时为[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 H 和 W 维度上均为该值。默认值：0。
    - **ceil_mode** (bool，可选) - 是否用 ceil 函数计算输出的 height 和 width，如果设置为 False，则使用 floor 函数来计算，默认为 False。
    - **data_format** (str，可选)：输入和输出的数据格式，可以是"NCL"和"NLC"。N 是批尺寸，C 是通道数，L 是特征长度。默认值："NCL"
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，长度），即 NCL 格式的 3-D Tensor。其数据类型为 float32 或 float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度），即 NCL 格式的 3-D Tensor。其数据类型与输入 x 相同。

返回
:::::::::
计算 LPPool1D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.LPPool1D
