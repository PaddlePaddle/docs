.. _cn_api_nn_functional_fold:

fold
-------------------------------

.. py:function:: paddle.nn.functional.fold(x, output_sizes, kernel_sizes, strides=1, paddings=0, dilations=1, name=None)


将一个滑动局部块组合成一个大的 Tensor。通常也被称为 col2im，用于批处理二维图像 Tensor。Fold 通过对所有包含块的值求和来计算结果中的每个大 Tensor 的组合值。


对于输入 x，如果形状为[N, C_in, L]，其输出形状[N, C_out, H_out, W_out]，计算过程如下：

.. math::

    H_{out} &= output\_size[0] \\
    W_{out} &= output\_size[1] \\
    C_{out} &= \frac{C_{in}}{kernel\_sizes[0]\times kernel\_sizes[1]} \\


参数
:::::::::
    - **x**  (Tensor) – 输入 3-D Tensor，形状为[N, C, L]，数据类型为 float32 、 float64 、 complex64 或 complex128
    - **output_sizes**  (int|list|tuple) – 输出尺寸，整数或者整型列表。如为列表类型应包含两个元素 ``[output_size_h, output_size_w]``。如果为整数 o，则输出形状会被认为 ``[o, o]``。
    - **kernel_size** (int|list|tuple) - 卷积核大小，整数或者整型列表。如为列表类型应包含两个元素 ``[k_h, k_w]``。如果为整数 k，则输出形状会被认为 ``[k, k]``。
    - **strides** (int|list|tuple，可选) - 步长大小，整数或者整型列表。如为列表类型应包含两个元素 ``[stride_h, stride_w]``。如果为整数 stride，则输出形状会被认为 ``[sride, stride]``。默认为[1,1]。
    - **paddings** (int|list|tuple，可选) – 每个维度的扩展，整数或者整型列表。如果为整型列表，长度应该为 4 或者 2；长度为 4 对应的 padding 参数是：[padding_top, padding_left，padding_bottom, padding_right]，长度为 2 对应的 padding 参数是[padding_h, padding_w]，会被当作[padding_h, padding_w, padding_h, padding_w]处理。如果为整数 padding，则会被当作[padding, padding, padding, padding]处理。默认值为 0。
    - **dilations** (int|list|tuple，可选) – 卷积膨胀，整型列表或者整数。如果为整型列表，应该包含两个元素[dilation_h, dilation_w]。如果是整数 dilation，会被当作整型列表[dilation, dilation]处理。默认值为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



形状
:::::::::
 - **输出** ：Tensor，fold 操作之后的结果，形状如上面所描述的[N, Cout, H_out, W_out]，数据类型与 ``x`` 相同


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.fold
