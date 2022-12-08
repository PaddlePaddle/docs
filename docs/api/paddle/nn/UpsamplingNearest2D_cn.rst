.. _cn_api_paddle_nn_UpsamplingNearest2D:

UpsamplingNearest2D
-------------------------------

.. py:function:: paddle.nn.UpsamplingNearest2D(size=None,scale_factor=None, data_format='NCHW',name=None)



调整一个 batch 中图片的大小。

输入为 4-D Tensor 时形状为(num_batches, channels, in_h, in_w)或者(num_batches, in_h, in_w, channels)，调整大小只适用于高度和宽度对应的维度。

支持的插值方法：

    NEAREST：最近邻插值


最近邻插值是在输入 Tensor 的高度和宽度上进行最近邻插值。


示例：

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


参数
::::::::::::

    - **size** (list|tuple|Tensor|None) - 输出 Tensor，输入为 4D Tensor，形状为(out_h, out_w)的 2-D Tensor。如果 :code:`size` 是列表，每一个元素可以是整数或者形状为[1]的变量。如果 ``size`` 是变量，则其维度大小为 1。默认值为 None。
    - **scale_factor** (float|Tensor|list|None)-输入的高度或宽度的乘数因子。``size`` 和 ``scale_factor`` 至少要设置一个。``size`` 的优先级高于 ``scale_factor``。默认值为 None。如果 ``scale_factor`` 是一个 list 或 tuple，它必须与输入的 shape 匹配。
    - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致。对于 4-D Tensor，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，默认值：'NCHW'。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。



代码示例
::::::::::::

COPY-FROM: paddle.nn.UpsamplingNearest2D
