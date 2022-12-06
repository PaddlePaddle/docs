.. _cn_api_fluid_layers_resize_nearest:

resize_nearest
-------------------------------

.. py:function:: paddle.fluid.layers.resize_nearest(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, data_format='NCHW')




该 OP 对输入图片进行大小调整，在高度方向宽度方向进行最邻近插值（nearest neighbor interpolation）操作。

输出形状按优先级顺序依据 ``actual_shape`` , ``out_shape`` 和 ``scale`` 而定。

**注意：** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

::

    Example:

          For scale:

            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)

            else:

              scale_factor = float(in_size/out_size)


          Nearest neighbor interpolation:

          if align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = \left \lfloor {H_{in} * scale_{}factor}} \right \rfloor
              W_out = \left \lfloor {W_{in} * scale_{}factor}} \right \rfloor

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

最邻近插值的详细介绍请参照：`Wiki Nearest-neighbor interpolation <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>`_


参数
::::::::::::

  - **input** (Variable) - 4-D Tensor，数据类型为 float32、float64 或 uint8，其数据格式由参数 ``data_format`` 指定。
  - **out_shape** (list|tuple|Variable|None) - 双线性插值法调整后的输出，维度为[out_h, out_w]的 2-D Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者 shape 为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为 1。默认值为 None。
  - **scale** (float|Variable|None) – 输入高宽的乘数因子。``out_shape`` 和 ``scale`` 二者至少设置其一。``out_shape`` 具有比 ``scale`` 更高的优先级。默认值为 None。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定 actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说，:code:`actual_shape` 具有最高的优先级。注意：如果希望动态指定输出形状，建议使用 :code:`out_shape`，因为 :code:`actual_shape` 未来将被弃用。在使用 actual_shape 指定输出形状时，仍然需要设置 out_shape 和 scale 之一，否则在图形构建阶段会出现错误。默认值为 None。
  - **align_corners** （bool）- 一个可选的 bool 型参数，如果为 True，则将输入和输出 Tensor 的 4 个角落像素的中心对齐，并保留角点像素的值。默认值为 True。
  - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。

返回
::::::::::::
4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。

返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.resize_nearest
