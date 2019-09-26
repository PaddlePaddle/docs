.. _cn_api_fluid_layers_resize_bilinear:

resize_bilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_bilinear(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, align_mode=1, data_format='NCHW')

**注意:** 参数 ``actual_shape`` 将被弃用，请使用 ``out_shape`` 替代。

该OP应用双向性插值法调整输入图片的大小，输出形状按优先级由actual_shape、out_shape和scale指定。

双线性插值是对线性插值的扩展,即二维变量方向上(如h方向和w方向)插值。关键思想是先在一个方向上执行线性插值，然后再在另一个方向上执行线性插值。

详情请参阅 `维基百科 <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_ 。

align_corners和align_mode是可选参数，插值的计算方法可以由它们选择。


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



参数:
    - **input** (Variable) - 4-D Tensor，数据类型为float32、float64或uint8，其数据格式由参数 ``data_format`` 指定。
    - **out_shape** (list|tuple|Variable|None) - 双线性层的输出形状，维度为[out_h, out_w]的二维Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者维度为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为1。默认值为None。
    - **scale** (float|Variable|None) - 用于输入高度或宽度的乘数因子。out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值为None。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。
    - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`out_shape` , 因为 :code:`out_shape` 未来将被弃用。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值为None。
    - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值为True
    - **align_mode** （int）- 双线性插值的可选项。 可以是'0'代表src_idx = scale *（dst_indx + 0.5）-0.5；如果为'1' ，代表src_idx = scale * dst_index。
    - **data_format** （str，可选）- 数据格式，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，默认值：'NCHW'。

返回：4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。

**代码示例**

.. code-block:: python
  
  import paddle.fluid as fluid
  input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
  # input.shape = [-1, 3, 6, 9], where -1 indicates batch size, and it will get the exact value in runtime.

  out0 = fluid.layers.resize_bilinear(input, out_shape=[12, 12])
  # out0.shape = [-1, 3, 12, 12], it means out0.shape[0] = input.shape[0] in runtime.

  # out_shape is a list in which each element is a integer or a tensor Variable
  dim1 = fluid.layers.data(name="dim1", shape=[1], dtype="int32", append_batch_size=False)
  out1 = fluid.layers.resize_bilinear(input, out_shape=[12, dim1])
  # out1.shape = [-1, 3, 12, -1]

  # out_shape is a 1-D tensor Variable
  shape_tensor = fluid.layers.data(name="shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
  out2 = fluid.layers.resize_bilinear(input, out_shape=shape_tensor)
  # out2.shape = [-1, 3, -1, -1]

  # when use actual_shape
  actual_shape_tensor = fluid.layers.data(name="actual_shape_tensor", shape=[2], dtype="int32", append_batch_size=False)
  out3 = fluid.layers.resize_bilinear(input, out_shape=[4, 4], actual_shape=actual_shape_tensor)
  # out3.shape = [-1, 3, 4, 4]

  # scale is a Variable
  scale_tensor = fluid.layers.data(name="scale", shape=[1], dtype="float32", append_batch_size=False)
  out4 = fluid.layers.resize_bilinear(input, scale=scale_tensor)
  # out4.shape = [-1, 3, -1, -1]
