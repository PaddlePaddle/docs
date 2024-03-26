.. _cn_api_paddle_sparse_nn_functional_max_pool3d:

max_pool3d
-------------------------------

.. py:function:: paddle.sparse.nn.functional.max_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, data_format="NDHWC", name=None)

该函数是一个三维最大池化函数，根据输入参数 `kernel_size` , `stride` , `padding` 等参数对输入 `x` 做最大池化操作。

参数
:::::::::
    - **x** (Tensor) - 形状为 [N,D,H,W, C] 的 5-D SparseCooTensor，N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度，数据类型为 float32 或 float64。
    - **kernel_size** (int|list|tuple) - 池化核大小。如果它是一个元组或列表，它必须包含三个整数值，(pool_size_Depth，pool_size_Height, pool_size_Width)。若为一个整数，则表示 D，H 和 W 维度上均为该值，比如若 kernel_size=2，则池化核大小为[2,2,2]。
    - **stride** (int|list|tuple，可选) - 池化层的步长。如果它是一个元组或列表，它将包含三个整数，(pool_stride_Depth，pool_stride_Height, pool_stride_Width)。若为一个整数，则表示 D, H 和 W 维度上 stride 均为该值。默认值为 kernel_size。
    - **padding** (string|int|list|tuple，可选) - 池化填充。如果它是一个字符串，可以是"VALID"或者"SAME"，表示填充算法。如果它是一个元组或列表，它可以有 3 种格式：

        - (1)包含 3 个整数值：[pad_depth, pad_height, pad_width]；
        - (2)包含 6 个整数值：[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]；
        - (3)包含 5 个二元组：当 data_format 为"NCDHW"时为[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]，当 data_format 为"NDHWC"时为[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]。若为一个整数，则表示 D、H 和 W 维度上均为该值。默认值：0
    - **ceil_mode** (bool，可选) - 是否用 ceil 函数计算输出高度和宽度。如果是 True，则使用 `ceil` 计算输出形状的大小。默认为 False
    - **data_format** (str，可选) - 输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，D 是特征深度，H 是特征高度，W 是特征宽度。当前只支持："NDHWC"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。



返回
:::::::::
5-D Tensor，数据类型与输入 x 一致。


代码示例
:::::::::

COPY-FROM: paddle.sparse.nn.functional.max_pool3d
