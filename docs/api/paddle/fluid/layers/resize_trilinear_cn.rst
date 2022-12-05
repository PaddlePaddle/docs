.. _cn_api_fluid_layers_resize_trilinear:

resize_trilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_trilinear(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, align_mode=1, data_format='NCDHW')




**注意：** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

该层对输入进行放缩，基于给定的由 ``actual_shape`` , ``out_shape`` , ``scale`` 确定的输出 shape，进行三线插值。三线插值是包含三个参数的线性插值方程（D 方向，H 方向，W 方向），在一个 3D 格子上进行三个方向的线性插值。更多细节，请参考维基百科：https://en.wikipedia.org/wiki/Trilinear_interpolation
Align_corners 和 align_mode 都是可选参数，可以用来设置插值的计算方法，如下：

::

    Example:

          For scale:

          if align_corners = True && out_size > 1 :

            scale_factor = (in_size-1.0)/(out_size-1.0)

          else:

            scale_factor = float(in_size/out_size)

          Bilinear interpolation:

            if:
                align_corners = False , align_mode = 0

                input : (N,C,D_in,H_in,W_in)
                output: (N,C,D_out,H_out,W_out) where:

                D_out = (D_{in}+0.5) * scale_{factor} - 0.5
                H_out = (H_{in}+0.5) * scale_{factor} - 0.5
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5


            else:

                input : (N,C,D_in,H_in,W_in)
                output: (N,C,D_out,H_out,W_out) where:

                D_out = D_{in} * scale_{factor}
                H_out = H_{in} * scale_{factor}
                W_out = W_{in} * scale_{factor}

参数
::::::::::::

  - **input** (Variable) – 5-D Tensor，数据类型为 float32、float64 或 uint8，其数据格式由参数 ``data_format`` 指定。
  - **out_shape** (list|tuple|Variable|None) – 调整最近邻层的输出形状，形式为(out_h, out_w)。默认值：None。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者 shape 为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为 1。
  - **scale** (float|None) – 输入高、宽的乘法器。``out_shape`` 和 ``scale`` 二者至少设置其一。``out_shape`` 具有比 ``scale`` 更高的优先级。默认：None
  - **name** (str|None) – 输出变量的命名
  - **actual_shape** (Variable) – 可选输入，动态设置输出 Tensor 的形状。如果提供该值，图片放缩会依据此形状进行，而非依据 ``out_shape`` 和 ``scale``。即为，``actual_shape`` 具有最高的优先级。如果想动态指明输出形状，推荐使用 ``out_shape``，因为 ``actual_shape`` 未来将被弃用。当使用 ``actual_shape`` 来指明输出形状，``out_shape`` 和 ``scale`` 也应该进行设置，否则在图形生成阶段将会报错。默认：None
  - **align_corners** （bool）- 一个可选的 bool 型参数，如果为 True，则将输入和输出 Tensor 的 4 个角落像素的中心对齐，并保留角点像素的值。默认值：True
  - **align_mode** (bool) - (int，默认为'1')，双线性插值选项，src_idx = scale*(dst_index+0.5)-0.5 时取'0'，src_idx = scale*dst_index 时取'1'。
  - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCDHW"。

返回
::::::::::::
5-D Tensor，形状为 (num_batches, channels, out_d, out_h, out_w) 或 (num_batches, out_d, out_h, out_w, channels)。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.resize_trilinear
