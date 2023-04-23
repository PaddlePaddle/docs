.. _cn_api_nn_AvgPool1D:


AvgPool1D
-------------------------------

.. py:function:: paddle.nn.AvgPool1D(kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None)

根据输入 `x` , `kernel_size` 等参数对一个输入 Tensor 计算 1D 的平均池化。输入和输出都是 3-D Tensor，
默认是以 `NCL` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`L` 是输入特征的长度。

假设输入形状是(N, C, L)，输出形状是 (N, C, L_{out})，卷积核尺寸是 k, 1d 平均池化计算公式如下：

..  math::

    Output(N_i, C_i, l) = mean(Input[N_i, C_i, stride \times l:stride \times l+k])

参数
:::::::::
    - **kernel_size** (int|list|tuple)：池化核的尺寸大小。如果 kernel_size 为 list 或 tuple 类型，其必须包含一个整数，最终池化核的大小为该数值。
    - **stride** (int|list|tuple，可选)：池化操作步长。如果 stride 为 list 或 tuple 类型，其必须包含一个整数，最终池化操作的步长为该数值。默认值为 None，这时会使用 kernel_size 作为 stride。
    - **padding** (str|int|list|tuple，可选)：池化补零的方式。如果 padding 是一个字符串，则必须为 `SAME` 或者 `VALID`。如果是 turple 或者 list 类型，则应是 `[pad_left, pad_right]` 形式。如果 padding 是一个非 0 值，那么表示会在输入的两端都 padding 上同样长度的 0。默认值为 0。
    - **exclusive** (bool，可选)：是否用额外 padding 的值计算平均池化结果，默认为 True。
    - **ceil_mode** (bool，可选)：是否用 ceil 函数计算输出的 height 和 width，如果设置为 False，则使用 floor 函数来计算，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，长度），即 NCL 格式的 3-D Tensor。其数据类型为 float32 或 float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度），即 NCL 格式的 3-D Tensor。其数据类型与输入 x 相同。

返回
:::::::::
计算 AvgPool1D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.AvgPool1D
