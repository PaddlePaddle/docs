.. _cn_api_paddle_compat_nn_functional_unfold:

unfold
-------------------------------

.. py:function:: paddle.compat.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

PyTorch 兼容的 im2col 操作，将二维滑动窗口覆盖的区域展开为列。

参数
:::::::::

    - **input** (Tensor) - NCHW 布局的四维输入 Tensor。
    - **kernel_size** (int|list|tuple|Tensor) - 卷积核大小。
    - **dilation** (int|list|tuple|Tensor，可选) - 卷积核膨胀系数。默认值：1。
    - **padding** (int|list|tuple|Tensor，可选) - 输入填充大小。默认值：0。
    - **stride** (int|list|tuple|Tensor，可选) - 滑动步长。默认值：1。

返回
:::::::::

Tensor，形状为 ``[N, C * kernel_size[0] * kernel_size[1], L]``。

代码示例
:::::::::

COPY-FROM: paddle.compat.nn.functional.unfold
