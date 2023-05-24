.. _cn_api_paddle_nn_UpsamplingBilinear2D:

UpsamplingBilinear2D
-------------------------------

.. py:function:: paddle.nn.UpsamplingBilinear2D(size=None,scale_factor=None, data_format='NCHW',name=None)



调整一个 batch 中图片的大小。

输入为 4-D Tensor 时形状为(num_batches, channels, in_h, in_w)或者(num_batches, in_h, in_w, channels)，调整大小只适用于高度和宽度对应的维度。

支持的插值方法：

    BILINEAR：双线性插值


双线性插值是线性插值的扩展，用于在直线 2D 网格上插值两个变量（例如，该操作中的 H 方向和 W 方向）的函数。关键思想是首先在一个方向上执行线性插值，然后在另一个方向上再次执行线性插值。

有关双线性插值的详细信息，请参阅维基百科：
https://en.wikipedia.org/wiki/Bilinear_interpolation


参数
::::::::::::

    - **size** (list|tuple|Tensor|None) - 输出 Tensor，输入为 4D Tensor，形状为为(out_h, out_w)的 2-D Tensor。如果 :code:`size` 是列表，每一个元素可以是整数或者形状为[]的 0-D Tensor。如果 :code:`size` 是 Tensor，则其为 1D Tensor。默认值为 None。
    - **scale_factor** (float|Tensor|list|tuple|None)-输入的高度或宽度的乘数因子。``size`` 和 ``scale_factor`` 至少要设置一个。``size`` 的优先级高于 ``scale_factor``。默认值为 None。如果 ``scale_factor`` 是一个 list 或 tuple，它必须与输入的 shape 匹配。
    - **data_format** （str，可选）- 指定输入的数据格式，输出的数据格式将与输入保持一致。对于 4-D Tensor，支持 NCHW(num_batches, channels, height, width) 或者 NHWC(num_batches, height, width, channels)，默认值：'NCHW'。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
4-D Tensor，形状为 (num_batches, channels, out_h, out_w) 或 (num_batches, out_h, out_w, channels)。



代码示例
::::::::::::

COPY-FROM: paddle.nn.UpsamplingBilinear2D
