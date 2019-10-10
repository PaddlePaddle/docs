.. _cn_api_fluid_layers_image_resize:

image_resize
-------------------------------

.. py:function:: paddle.fluid.layers.image_resize(input, out_shape=None, scale=None, name=None, resample='BILINEAR', actual_shape=None, align_corners=True, align_mode=1, data_format='NCHW')

**注意:** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

该OP用于调整一个batch中图片的大小。

输入为4-D Tensor时形状为(num_batches, channels, in_h, in_w)或者(num_batches, in_h, in_w, channels)，输入为5-D Tensor时形状为(num_batches, channels, in_d, in_h, in_w)或者(num_batches, in_d, in_h, in_w, channels)，并且调整大小只适用于深度，高度和宽度对应的维度。

支持的插值方法:

    BILINEAR：双线性插值

    TRALINEAR：三线插值

    NEAREST：最近邻插值


最近邻插值是在输入张量的高度和宽度上进行最近邻插值。

双线性插值是线性插值的扩展，用于在直线2D网格上插值两个变量（例如，该操作中的H方向和W方向）的函数。 关键思想是首先在一个方向上执行线性插值，然后在另一个方向上再次执行线性插值。

三线插值是线性插值的一种扩展，是3参数的插值方程（比如op里的D,H,W方向），在三个方向上进行线性插值。

Align_corners和align_mode是可选参数，插值的计算方法可以由它们选择。

示例:

::

      For scale:

        if align_corners = True && out_size > 1 :

          scale_factor = (in_size-1.0)/(out_size-1.0)

        else:

          scale_factor = float(in_size/out_size)


      Nearest neighbor interpolation:

      if:
          align_corners = False

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

      Bilinear interpolation:

      if:
          align_corners = False , align_mode = 0

          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = (H_{in}+0.5) * scale_{factor} - 0.5
          W_out = (W_{in}+0.5) * scale_{factor} - 0.5


      else:

          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = H_{in} * scale_{factor}
          W_out = W_{in} * scale_{factor}

      Trilinear interpolation:

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


有关最近邻插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

有关双线性插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Bilinear_interpolation

有关三线插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Trilinear_interpolation

参数:
    - **input** (Variable) - 4-D或5-D Tensor，数据类型为float32、float64或uint8，其数据格式由参数 ``data_format`` 指定。
    - **out_shape** (list|tuple|Variable|None) - 输出Tensor，输入为4D张量时，形状为为(out_h, out_w)的2-D Tensor。输入为5-D Tensor时，形状为(out_d, out_h, out_w)的3-D Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者形状为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为1。默认值为None。
    - **scale** (float|Variable|None)-输入的高度或宽度的乘数因子 。 out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值为None。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。
    - **resample** (str) - 插值方法。支持“双线性”,“三线性”,“临近插值”。默认值为双线性插值。
    - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`out_shape` ，因为 :code:`actual_shape` 未来将被弃用。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值:None
    - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值为True
    - **align_mode** （int）- 双线性插值的可选项。 可以是 '0' 代表src_idx = scale *（dst_indx + 0.5）-0.5；如果为'1' ，代表src_idx = scale * dst_index。
    - **data_format** （str，可选）- 对于4-D Tensor，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，对于5-D Tensor，支持 NCDHW(num_batches, channels, depth, height, width)或者 NDHWC(num_batches, depth, height, width, channels)，默认值：'NCHW'。

返回：4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)；或者5-D Tensor，形状为 (num_batches, channels, out_d, out_h, out_w) 或 (num_batches, out_d, out_h, out_w, channels)。

返回类型: 变量（variable）

抛出异常：
    - :code:`TypeError` - out_shape应该是一个列表、元组或变量。
    - :code:`TypeError` - actual_shape应该是变量或None。
    - :code:`ValueError` - image_resize的"resample"只能是"BILINEAR"或"TRILINEAR"或"NEAREST"。
    - :code:`ValueError` - out_shape 和 scale 不可同时为 None。
    - :code:`ValueError` - out_shape 的长度必须为2如果输入是4D张量。
    - :code:`ValueError` - out_shape 的长度必须为3如果输入是5D张量。
    - :code:`ValueError` - scale应大于0。
    - :code:`TypeError`  - align_corners 应为bool型。
    - :code:`ValueError` - align_mode 只能取 ‘0’ 或 ‘1’。
    - :code:`ValueError` - data_format 只能取 ‘NCHW’、‘NHWC’、‘NCDHW’ 或者 ‘NDHWC’。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
  # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

  out = fluid.layers.image_resize(input, out_shape=[12, 12], resample="NEAREST")
  out0 = fluid.layers.image_resize(input, out_shape=[12, 12], resample="NEAREST")
  # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

  # out_shape is a list in which each element is a integer or a tensor Variable
  dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
  out1 = fluid.layers.image_resize(input, out_shape=[12, dim1], resample="NEAREST")
  # out1.shape = [-1, 3, 12, -1]

  # out_shape is a 1-D tensor Variable
  shape_tensor = fluid.layers.data(name="shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
  out2 = fluid.layers.image_resize(input, out_shape=shape_tensor, resample="NEAREST")
  # out2.shape = [-1, 3, -1, -1]

  # when use actual_shape
  actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
  out3 = fluid.layers.image_resize(input, out_shape=[4, 4], resample="NEAREST", actual_shape=actual_shape_tensor)
  # out3.shape = [-1, 3, 4, 4]

  # scale is a Variable
  scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
  out4 = fluid.layers.image_resize(input, scale=scale_tensor)
  # out4.shape = [-1, 3, -1, -1]










