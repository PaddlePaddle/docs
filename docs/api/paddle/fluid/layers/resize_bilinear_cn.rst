.. _cn_api_fluid_layers_resize_bilinear:

resize_bilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_bilinear(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, align_mode=1, data_format='NCHW')




**注意：** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

该 OP 应用双向性插值法调整输入图片的大小，输出形状按优先级由 actual_shape、out_shape 和 scale 指定。

双线性插值是对线性插值的扩展，即二维变量方向上(如 h 方向和 w 方向)插值。关键思想是先在一个方向上执行线性插值，然后再在另一个方向上执行线性插值。

详情请参阅 `维基百科 <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_ 。

align_corners 和 align_mode 是可选参数，插值的计算方法可以由它们选择。


::

    Example:

      For scale:

        if align_corners = True && out_size > 1 :

          scale_factor = (in_size-1.0)/(out_size-1.0)

        else:

          scale_factor = float(in_size/out_size)

    Bilinear interpolation:

      if align_corners = False , align_mode = 0

          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = (H_{in}+0.5) * scale_{factor} - 0.5
          W_out = (W_{in}+0.5) * scale_{factor} - 0.5


      else:

          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = H_{in} * scale_{factor}
          W_out = W_{in} * scale_{factor}



参数
::::::::::::

    - **input** (Variable) - 4-D Tensor，数据类型为 float32、float64 或 uint8，其数据格式由参数 ``data_format`` 指定。
    - **out_shape** (list|tuple|Variable|None) - 双线性层的输出形状，维度为[out_h, out_w]的二维 Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者维度为[]的 Tensor。如果 :code:`out_shape` 是 Tensor，则其维度大小为 1。默认值为 None。
    - **scale** (float|Variable|None) - 用于输入高度或宽度的乘数因子。out_shape 和 scale 至少要设置一个。out_shape 的优先级高于 scale。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定 actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说，:code:`actual_shape` 具有最高的优先级。注意：如果希望动态指定输出形状，建议使用 :code:`out_shape`，因为 :code:`actual_shape` 未来将被弃用。在使用 actual_shape 指定输出形状时，仍然需要设置 out_shape 和 scale 之一，否则在图形构建阶段会出现错误。默认值为 None。
    - **align_corners** （bool）- 一个可选的 bool 型参数，如果为 True，则将输入和输出 Tensor 的 4 个角落像素的中心对齐，并保留角点像素的值。默认值为 True
    - **align_mode** （int）- 双线性插值的可选项。可以是'0'代表 src_idx = scale *（dst_indx + 0.5）-0.5；如果为'1'，代表 src_idx = scale * dst_index。
    - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。

返回
::::::::::::
4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.resize_bilinear
