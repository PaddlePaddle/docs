.. _cn_api_fluid_layers_resize_nearest:

resize_nearest
-------------------------------

.. py:function:: paddle.fluid.layers.resize_nearest(input, out_shape=None, scale=None, name=None, actual_shape=None, align_corners=True)

该层对输入进行放缩，在第三维（高度方向）和第四维（宽度方向）进行最邻近插值（nearest neighbor interpolation）操作。
输出形状按优先级顺序依据 ``actual_shape`` , ``out_shape`` 和 ``scale`` 而定。

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
  - **input** (Variable) – 插值运算的输入张量, 是一个形为 (N,C,H,W) 的四维张量
  - **out_shape** (list|tuple|Variable|None) – 调整最近邻层的输出形状，形式为(out_h, out_w)。默认值：None。
  - **scale** (float|None) – 输入高、宽的乘法器。 ``out_shape`` 和 ``scale`` 二者至少设置其一。 ``out_shape`` 具有比 ``scale`` 更高的优先级。 默认: None
  - **name** (str|None) – 输出变量的命名
  - **actual_shape** (Variable) – 可选输入， 动态设置输出张量的形状。 如果提供该值， 图片放缩会依据此形状进行， 而非依据 ``out_shape`` 和 ``scale`` 。 即为， ``actual_shape`` 具有最高的优先级。 如果想动态指明输出形状，推荐使用 ``actual_shape`` 取代 ``out_shape`` 。 当使用 ``actual_shape`` 来指明输出形状， ``out_shape`` 和 ``scale`` 也应该进行设置, 否则在图形生成阶段将会报错。默认: None
  - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值：True

返回：插值运算的输出张量，是一个形为 [N,C,H,W] 的四维张量

**代码示例**

..  code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9], dtype="float32")
    out = fluid.layers.resize_nearest(input, out_shape=[12, 12])










