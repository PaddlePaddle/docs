.. _cn_api_paddle_nn_UpSample:

Upsample
-------------------------------

.. py:function:: paddle.nn.Upsample(x,size=None,scale_factor=None,mode='nearest',align_corners=False,align_mode=0,data_format='NCHW',name=None):



该OP用于调整一个batch中图片的大小。

输入为4-D Tensor时形状为(num_batches, channels, in_h, in_w)或者(num_batches, in_h, in_w, channels)，输入为5-D Tensor时形状为(num_batches, channels, in_d, in_h, in_w)或者(num_batches, in_d, in_h, in_w, channels)，并且调整大小只适用于深度，高度和宽度对应的维度。

支持的插值方法:

    NEAREST：最近邻插值

    BILINEAR：双线性插值

    TRALINEAR：三线性插值

    BICUBIC：双三次插值

    LINEAR: 线性插值

    AREA: 面积插值



最近邻插值是在输入张量的高度和宽度上进行最近邻插值。

双线性插值是线性插值的扩展，用于在直线2D网格上插值两个变量（例如，该操作中的H方向和W方向）的函数。 关键思想是首先在一个方向上执行线性插值，然后在另一个方向上再次执行线性插值。

三线插值是线性插值的一种扩展，是3参数的插值方程（比如op里的D,H,W方向），在三个方向上进行线性插值。

双三次插值是在二维网格上对数据点进行插值的三次插值的扩展，它能创造出比双线性和最近临插值更为光滑的图像边缘。

Align_corners和align_mode是可选参数，插值的计算方法可以由它们选择。

示例:

::

      
      scale 计算方法：

        if align_corners = True && out_size > 1 :

          scale_factor = (in_size-1.0)/(out_size-1.0)

        else:

          scale_factor = float(in_size/out_size)


      不同插值方式的输出纬度计算规则：

      Nearest neighbor interpolation:


          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = \left \lfloor {H_{in} * scale_{}factor}} \right \rfloor
          W_out = \left \lfloor {W_{in} * scale_{}factor}} \right \rfloor


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

      Bicubic interpolation:

      if:
          align_corners = False

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

有关双三次插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Bicubic_interpolation

参数:
    - **x** (Tensor) - 4-D或5-D Tensor，数据类型为float32、float64或uint8，其数据格式由参数 ``data_format`` 指定。
    - **size** (list|tuple|Tensor|None) - 输出Tensor，输入为4D张量时，形状为为(out_h, out_w)的2-D Tensor。输入为5-D Tensor时，形状为(out_d, out_h, out_w)的3-D Tensor。如果 :code:`out_shape` 是列表，每一个元素可以是整数或者形状为[1]的变量。如果 :code:`out_shape` 是变量，则其维度大小为1。默认值为None
    - **scale_factor** (float|Tensor|list|tuple|None)-输入的高度或宽度的乘数因子 。 out_shape和scale至少要设置一个。out_shape的优先级高于scale。默认值为None。
如果scale_factor是一个list或tuple，它必须与输入的shape匹配。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。
    - **mode** (str) - 插值方法。支持“双线性”,“三线性”,“临近插值”,"双三次","线性"，"面积"。默认值为最近临插值。
    - **align_corners** （bool）- 一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。 默认值为True
    - **align_mode** （int）- 双线性插值的可选项。 可以是 '0' 代表src_idx = scale *（dst_indx + 0.5）-0.5；如果为'1' ，代表src_idx = scale * dst_index。
    - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致。对于4-D Tensor，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，对于5-D Tensor，支持 NCDHW(num_batches, channels, depth, height, width)或者 NDHWC(num_batches, depth, height, width, channels)，默认值：'NCHW'。

返回：4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)；或者5-D Tensor，形状为 (num_batches, channels, out_d, out_h, out_w) 或 (num_batches, out_d, out_h, out_w, channels)。

返回类型: 变量（variable）

抛出异常：
    - :code:`TypeError` - out_shape应该是一个列表、元组或张量。
    - :code:`TypeError` - actual_shape应该是变量或None。
    - :code:`ValueError` - Upsample的"mode"只能是"BILINEAR"或"TRILINEAR"或"NEAREST"或"BICUBIC"或"LINEAR"或"AREA"。
    - :code:`ValueError` - 'linear' 只支持 3-D tensor。
    - :code:`ValueError` - 'bilinear', 'bicubic' ，'nearest' 只支持 4-D tensor。
    - :code:`ValueError` - 'trilinear' 只支持 5-D tensor
    - :code:`ValueError` - out_shape 和 scale 不可同时为 None。
    - :code:`ValueError` - out_shape 的长度必须为1如果输入是3D张量
    - :code:`ValueError` - out_shape 的长度必须为2如果输入是4D张量。
    - :code:`ValueError` - out_shape 的长度必须为3如果输入是5D张量。
    - :code:`ValueError` - scale应大于0。
    - :code:`TypeError`  - align_corners 应为bool型。
    - :code:`ValueError` - align_mode 只能取 ‘0’ 或 ‘1’。
    - :code:`ValueError` - data_format 只能取 'NCW', 'NCHW'、'NHWC'、'NCDHW' 或者 'NDHWC'。

**代码示例**

..  code-block:: python

        import paddle
        import paddle.nn as nn
        import numpy as np
        paddle.disable_static()
        input_data = np.random.rand(2,3,6,10).astype("float32")
        upsample_out  = paddle.nn.UpSample(size=[12,12])
        input = paddle.to_tensor(input_data)
        output = upsample_out(x=input)
        print(output.shape)
        # [2L, 3L, 12L, 12L]
