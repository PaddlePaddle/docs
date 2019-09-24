.. _cn_api_fluid_layers_resize_trilinear:

resize_trilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_trilinear(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, align_mode=1)

**注意:** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

该层对输入进行放缩，基于给定的由 ``actual_shape`` , ``out_shape`` , ``scale`` 确定的输出shape，进行三线插值。三线插值是包含三个参数的线性插值方程（D方向，H方向， W方向）,在一个3D格子上进行三个方向的线性插值。更多细节，请参考维基百科：https://en.wikipedia.org/wiki/Trilinear_interpolation
Align_corners和align_mode都是可选参数，可以用来设置插值的计算方法，如下：

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

参数:
  - **input** (Variable) – 输入是shape为(num_batches, channels, in_d, in_h, in_w)的5-D张量。
  - **out_shape** (list|tuple|Variable|None) – 调整最近邻层的输出形状，形式为(out_h, out_w)。默认值：None。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者shape为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为1。
  - **scale** (float|None) – 输入高、宽的乘法器。 ``out_shape`` 和 ``scale`` 二者至少设置其一。 ``out_shape`` 具有比 ``scale`` 更高的优先级。 默认: None
  - **name** (str|None) – 输出变量的命名
  - **actual_shape** (Variable) – 可选输入， 动态设置输出张量的形状。 如果提供该值， 图片放缩会依据此形状进行， 而非依据 ``out_shape`` 和 ``scale`` 。 即为， ``actual_shape`` 具有最高的优先级。 如果想动态指明输出形状，推荐使用 ``out_shape`` ，因为 ``actual_shape`` 未来将被弃用。 当使用 ``actual_shape`` 来指明输出形状， ``out_shape`` 和 ``scale`` 也应该进行设置, 否则在图形生成阶段将会报错。默认: None
  - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值：True
  - **align_mode** (bool) - (int,默认为'1')，双线性插值选项，src_idx = scale*(dst_index+0.5)-0.5时取'0'，src_idx = scale*dst_index时取'1'。

返回：形为(num_batches, channels, out_d, out_h, out_w)的5-D张量

**代码示例**

..  code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9,11], dtype="float32")
    # input.shape = [-1, 3, 6, 9, 11], where -1 indicates batch size, and it will get the exact value in runtime.

    out0 = fluid.layers.resize_trilinear(input, out_shape=[12, 12, 12])
    # out0.shape = [-1, 3, 12, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

    # out_shape is a list in which each element is a integer or a tensor Variable
    dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
    out1 = fluid.layers.resize_trilinear(input, out_shape=[12, dim1, 4])
    # out1.shape = [-1, 3, 12, -1, 4]

    # out_shape is a 1-D tensor Variable
    shape_tensor = fluid.layers.data(name="shape_tensor", shape=[3], dtype="int32", append_batch_size=False)
    out2 = fluid.layers.resize_trilinear(input, out_shape=shape_tensor)
    # out2.shape = [-1, 3, -1, -1, -1]

    # when use actual_shape
    actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[3], dtype="int32", append_batch_size=False)
    out3 = fluid.layers.resize_trilinear(input, out_shape=[4, 4, 8], actual_shape=actual_shape_tensor)
    # out3.shape = [-1, 3, 4, 4, 8]

    # scale is a Variable
    scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
    out4 = fluid.layers.resize_trilinear(input, scale=scale_tensor)
    # out4.shape = [-1, 3, -1, -1, -1]
