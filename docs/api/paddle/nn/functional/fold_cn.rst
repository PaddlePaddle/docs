.. _cn_api_fluid_layers_fold:

fold
-------------------------------

.. py:function:: paddle.nn.functional.fold(x, output_sizes, kernel_sizes, strides=1, paddings=0, dilations=1, name=None)


该Op用于将一个滑动局部块组合成一个大的张量。通常也被称为col2im，用于批处理二维图像张量。Fold通过对所有包含块的值求和来计算结果中的每个大张量的组合值。


对于输入x，如果形状为[N, C_in, L]，其输出形状[N, C_out, H_out, W_out]， 计算过程如下:
.. math::

    H_out &=  output_size[0]
    W_out &=  output_size[1]
    C_out &=  C_in / kernel\_sizes[0] / kernel\_sizes[1]


参数：
    - **x**  (Tensor) – 输入3-D Tensor，形状为[N, C, L]，数据类型为float32或者float64
    - **output_sizes**  (int|list|tuple) – 输出尺寸，整数或者整型列表。如为列表类型应包含两个元素 ``[output_size_h, output_size_w]`` 。如果为整数o，则输出形状会被认为 ``[o, o]``。
    - **kernel_size** (int|list|tuple) - 卷积核大小，整数或者整型列表。如为列表类型应包含两个元素 ``[k_h, k_w]`` 。如果为整数k，则输出形状会被认为 ``[k, k]``。
    - **strides** (int|list|tuple, 可选) - 步长大小，整数或者整型列表。如为列表类型应包含两个元素 ``[stride_h, stride_w]`` 。如果为整数stride，则输出形状会被认为 ``[sride, stride]``。默认为[1,1]
    - **paddings** (int|list|tuple，可选) – 每个维度的扩展, 整数或者整型列表。如果为整型列表，长度应该为4或者2；长度为4 对应的padding参数是：[padding_top, padding_left，padding_bottom, padding_right]，长度为2对应的padding参数是[padding_h, padding_w]，会被当作[padding_h, padding_w, padding_h, padding_w]处理。如果为整数padding，则会被当作[padding, padding, padding, padding]处理。默认值为0
    - **dilations** (int|list|tuple，可选) – 卷积膨胀，整型列表或者整数。如果为整型列表，应该包含两个元素[dilation_h, dilation_w]。如果是整数dilation，会被当作整型列表[dilation, dilation]处理。默认值为1
    - **name** (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。



返回：Tensor,  fold操作之后的结果，形状如上面所描述的[N, Cout, H_out, W_out]，数据类型与 ``x`` 相同


**代码示例**:

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
    x = paddle.randn([2,3*2*2,12])
    y = F.fold(x, output_sizes=[4, 5], kernel_sizes=2)
    # y.shape = [2,3,4,5]





