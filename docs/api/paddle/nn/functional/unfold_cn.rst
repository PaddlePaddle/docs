.. _cn_api_paddle_nn_functional_unfold:

unfold
-------------------------------

.. py:function:: paddle.nn.functional.unfold(x, kernel_sizes, strides=1, paddings=0, dilation=1, name=None)




实现的功能与卷积中用到的 im2col 函数一样，通常也被称作为 im2col 过程。对于每一个卷积核覆盖下的区域，元素会被重新排成一列。当卷积核在整个图片上滑动时，将会形成一系列的列向量。对于每一个输入形状为[N, C, H, W]的 ``x``，都将会按照下面公式计算出一个形状为[N, Cout, Lout]的输出。


..  math::

       dkernel[0] &= dilations[0] * (kernel\_sizes[0] - 1) + 1

       dkernel[1] &= dilations[1] * (kernel\_sizes[1] - 1) + 1

       hout &= \frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

       wout &= \frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

       Cout &= C * kernel\_sizes[0] * kernel\_sizes[1]

       Lout &= hout * wout

**样例**：

::

      Given:
        x.shape = [5, 10, 25, 25]
        kernel_sizes = [3, 3]
        strides = 1
        paddings = 1

      Return:
        out.shape = [5, 90, 625]


参数
::::::::::::

    - **x**  (Tensor) – 输入 4-D Tensor，形状为[N, C, H, W]，数据类型为 float32 或者 float64
    - **kernel_sizes**  (int|list(tuple) of int) – 卷积核的尺寸，整数或者整型列表。如果为整型列表，应包含两个元素 ``[k_h, k_w]``，卷积核大小为 ``k_h * k_w``；如果为整数 k，会被当作整型列表 ``[k, k]`` 处理
    - **strides**  (int|list(tuple) of int，可选) – 卷积步长，整数或者整型列表。如果为整型列表，应该包含两个元素 ``[stride_h, stride_w]``。如果为整数，则 ``stride_h = stride_w = strides``。默认值为 1
    - **paddings** (int|list(tuple) of int，可选) – 每个维度的扩展，整数或者整型列表。如果为整型列表，长度应该为 4 或者 2；长度为 4 对应的 padding 参数是：[padding_top, padding_left，padding_bottom, padding_right]，长度为 2 对应的 padding 参数是[padding_h, padding_w]，会被当作[padding_h, padding_w, padding_h, padding_w]处理。如果为整数 padding，则会被当作[padding, padding, padding, padding]处理。默认值为 0
    - **dilations** (int|list(tuple) of int，可选) – 卷积膨胀，整型列表或者整数。如果为整型列表，应该包含两个元素[dilation_h, dilation_w]。如果是整数 dilation，会被当作整型列表[dilation, dilation]处理。默认值为 1
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
Tensor,  unfold 操作之后的结果，形状如上面所描述的[N, Cout, Lout]，Cout 每一个滑动 block 里面覆盖的元素个数，Lout 是滑动 block 的个数，数据类型与 ``x`` 相同


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.unfold
