.. _cn_api_fluid_layers_resize_bilinear:

resize_bilinear
-------------------------------

.. py:function:: paddle.fluid.layers.resize_bilinear(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True, align_mode=1)


根据指定的out_shape执行双线性插值调整输入大小，输出形状按优先级由actual_shape、out_shape和scale指定。

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
    - **input** (Variable) - 双线性插值的输入张量，是一个shape为(N x C x h x w)的4d张量。
    - **out_shape** (list|tuple|Variable|None) - 调整双线性层的输出形状，形式为(out_h, out_w)。默认值：None。
    - **scale** (float|None) - 用于输入高度或宽度的乘数因子。out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值：None。
    - **name** (str|None) - 输出变量名。
    - **actual_shape** (Variable) - 可选输入，用于动态指定输出形状。如果指定actual_shape，图像将根据给定的形状调整大小，而不是根据指定形状的 :code:`out_shape` 和 :code:`scale` 进行调整。也就是说， :code:`actual_shape` 具有最高的优先级。如果希望动态指定输出形状，建议使用 :code:`actual_shape` 而不是 :code:`out_shape` 。在使用actual_shape指定输出形状时，还需要设置out_shape和scale之一，否则在图形构建阶段会出现错误。默认值:None
    - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值：True
    - **align_mode** （int）- 双线性插值的可选项。 可以是'0'代表src_idx = scale *（dst_indx + 0.5）-0.5；可以为'1' ，代表src_idx = scale * dst_index。


返回： 插值运算的输出张量，其各维度是(N x C x out_h x out_w)


**代码示例**

.. code-block:: python
  
  import paddle.fluid as fluid
  input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
  out = fluid.layers.resize_bilinear(input, out_shape=[12, 12])








