.. _cn_api_paddle_compat_nn_AvgPool2d:

AvgPool2d
-------------------------------

.. py:class:: paddle.compat.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

PyTorch 兼容的二维平均池化层。输入和输出采用 NCHW 布局。

参数
:::::::::

    - **kernel_size** (int|list|tuple) - 池化核大小。
    - **stride** (int|list|tuple|None，可选) - 池化步长。默认值：None，使用 ``kernel_size``。
    - **padding** (str|int|list|tuple，可选) - 填充大小。默认值：0。
    - **ceil_mode** (bool，可选) - 是否使用 ceil 计算输出尺寸。默认值：False。
    - **count_include_pad** (bool，可选) - 计算平均值时是否计入填充元素。默认值：True。
    - **divisor_override** (int|None，可选) - 指定平均值计算的除数。默认值：None。
