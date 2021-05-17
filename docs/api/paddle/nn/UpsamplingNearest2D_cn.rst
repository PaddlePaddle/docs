.. _cn_api_paddle_nn_UpsamplingNearest2D:

UpsamplingNearest2D
-------------------------------

.. py:function:: paddle.nn.UpsamplingNearest2D(size=None,scale_factor=None, data_format='NCHW',name=None):



该OP用于调整一个batch中图片的大小。

输入为4-D Tensor时形状为(num_batches, channels, in_h, in_w)或者(num_batches, in_h, in_w, channels), 调整大小只适用于高度和宽度对应的维度。

支持的插值方法:

    NEAREST：最近邻插值


最近邻插值是在输入张量的高度和宽度上进行最近邻插值。


示例:

::

      
      scale 计算方法：

        if align_corners = True && out_size > 1 :

          scale_factor = (in_size-1.0)/(out_size-1.0)

        else:

          scale_factor = float(in_size/out_size)


      插值方式的输出纬度计算规则：

      Nearest neighbor interpolation:


          input : (N,C,H_in,W_in)
          output: (N,C,H_out,W_out) where:

          H_out = \left \lfloor {H_{in} * scale_{}factor}} \right \rfloor
          W_out = \left \lfloor {W_{in} * scale_{}factor}} \right \rfloor

有关最近邻插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation


参数:
    - **size** (list|tuple|Tensor|None) - 输出Tensor，输入为4D张量，形状为(out_h, out_w)的2-D Tensor。如果 :code:`size` 是列表，每一个元素可以是整数或者形状为[1]的变量。如果 ``size`` 是变量，则其维度大小为1。默认值为None。
    - **scale_factor** (float|Tensor|list|None)-输入的高度或宽度的乘数因子。 ``size`` 和 ``scale_factor`` 至少要设置一个。 ``size`` 的优先级高于 ``scale_factor`` 。默认值为None。如果 ``scale_factor`` 是一个list或tuple，它必须与输入的shape匹配。
    - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致。对于4-D Tensor，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，默认值：'NCHW'。
    - **name** (str|None, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

返回：4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。


抛出异常：
    - :code:`TypeError` - out_shape应该是一个列表、元组或变量。
    - :code:`ValueError` - out_shape 和 scale 不可同时为 None。
    - :code:`ValueError` - out_shape 的长度必须为2如果输入是4D张量。
    - :code:`ValueError` - scale应大于0。
    - :code:`ValueError` - data_format 只能取 ‘NCHW’、‘NHWC’


**代码示例**

..  code-block:: python

       import paddle
       import paddle.nn as nn

       input_data = paddle.rand(shape=(2,3,6,10))
       upsample_out  = paddle.nn.UpsamplingNearest2D(size=[12,12])
       output = upsample_out(input_data)
       print(output.shape)
       # [2L, 3L, 12L, 12L]
