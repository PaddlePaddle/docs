.. _cn_api_paddle_sparse_nn_MaxPool3D:

MaxPool3D
-------------------------------

.. py:class:: paddle.sparse.nn.MaxPool3D(kernel_size, stride=None, padding=0, return_mask=False, ceil_mode=False, data_format="NDHWC", name=None)

构建 ``MaxPool3D`` 类的一个可调用对象，其将构建一个三维最大池化层，根据输入参数 ``kernel_size``, ``stride``,
``padding`` 等参数对稀疏输入特征做最大池化操作。输入输出都是 "NDHWC" 格式，其中 N 是批大小，C 是特征的通道数，D、H、W 分别是特征的深度、高和宽。

参数
:::::::::
    - **kernel_size** (int|list|tuple) - 池化核大小。如果它是一个元组或列表，它必须包含三个整数值，(pool_size_Depth，pool_size_Height, pool_size_Width)。若为一个整数，则表示 D，H 和 W 维度上均为该值，比如若 kernel_size=2，则池化核大小为[2,2,2]。
    - **stride** (int|list|tuple，可选) - 池化层的步长。如果它是一个元组或列表，它将包含三个整数，(pool_stride_Depth，pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 D, H 和 W 维度上 stride 均为该值。默认值为 None ，这时会使用 kernel_size 作为 stride 。
    - **padding** (str|int|list|tuple，可选) - 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有 3 种格式：(1)包含 3 个整数值：[pad_depth, pad_height, pad_width]；(2)包含 6 个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；(3)包含 5 个二元组：``[[0, 0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]]``。若为一个整数，则表示 D、H 和 W 维度上均为该值。默认值：0 。
    - **return_mask** (bool，可选) - 当前该参数不会传递给底层池化函数，调用时始终返回单个池化结果 Tensor。默认值为 False 。
    - **ceil_mode** (bool，可选) - 是否用 ceil 函数计算输出深度、高度和宽度。如果是 True ，则使用 ``ceil`` 计算输出形状的大小。默认为 False 。
    - **data_format** (str，可选) - 输入和输出的数据格式。当前仅支持 "NDHWC"。N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None 。


形状
:::::::::
    - **x** (SparseCooTensor)：形状为 ``[N, D, H, W, C]`` 的 NDHWC 格式 5-D SparseCooTensor。数据类型为 float32 或 float64。
    - **output** (SparseCooTensor)：形状为 ``[N, D_out, H_out, W_out, C]`` 的 NDHWC 格式 5-D SparseCooTensor。其数据类型与输入相同。


返回
:::::::::
计算 MaxPool3D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.MaxPool3D
