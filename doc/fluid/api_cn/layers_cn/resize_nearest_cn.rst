.. _cn_api_fluid_layers_resize_nearest:

resize_nearest
-------------------------------

.. py:function:: paddle.fluid.layers.resize_nearest(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True)

该OP对输入图片进行大小调整，在第三维（高度方向）和第四维（宽度方向）进行最邻近插值（nearest neighbor interpolation）操作。
输出形状按优先级顺序依据 ``actual_shape`` , ``out_shape`` 和 ``scale`` 而定。

**注意:** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

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

最邻近插值的详细介绍请参照： `Wiki Nearest-neighbor interpolation <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>`_


参数:
  - **input** (Variable) - 输入维度为[num_batches, channels, in_h, in_w]的4-D Tensor。
  - **out_shape** (list|tuple|Variable|None) - 双线性插值法调整后的输出，维度为[out_h, out_w]的2-D Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者shape为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为1。默认值为None。
  - **scale** (float|Variable|None) – 输入高宽的乘数因子。 ``out_shape`` 和 ``scale`` 二者至少设置其一。 ``out_shape`` 具有比 ``scale`` 更高的优先级。 默认值为None。
  - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。
  - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`out_shape` , 因为 :code:`out_shape` 未来将被弃用。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值为None。
  - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值为True。

返回：维度为[num_batches, channels, out_h, out_w]的4-D Tensor。

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
    # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

    out0 = fluid.layers.resize_nearest(input, out_shape=[12, 12])
    # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

    # out_shape is a list in which each element is a integer or a tensor Variable
    dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
    out1 = fluid.layers.resize_nearest(input, out_shape=[12, dim1])
    # out1.shape = [-1, 3, 12, -1]

    # out_shape is a 1-D tensor Variable
    shape_tensor = fluid.layers.data(name="resize_shape", shape=[2], dtype="int32", append_batch_size=False)
    out2 = fluid.layers.resize_nearest(input, out_shape=shape_tensor)
    # out2.shape = [-1, 3, -1, -1]

    # when use actual_shape
    actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
    out3 = fluid.layers.resize_nearest(input, out_shape=[4, 4], actual_shape=actual_shape_tensor)
    # out3.shape = [-1, 3, 4, 4]

    # scale is a Variable
    scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
    out4 = fluid.layers.resize_nearest(input, scale=scale_tensor)
    # out4.shape = [-1, 3, -1, -1]











