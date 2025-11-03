.. _cn_api_paddle_nn_CircularPad2D:

CircularPad2D
-------------------------------
.. py:class:: paddle.nn.CircularPad2D(padding, data_format="NCHW", name=None)

使用循环填充的方式来填充输入张量的边界。

参数
::::::::::::

  - **padding** (Tensor|Sequence[int]|int) - 填充大小。如果 `padding` 是一个整数，则在左、右、上、下四个边界都应用相同的填充。如果 `padding` 是一个包含四个整数的列表或元组，它将被解析为 `(pad_left, pad_right, pad_top, pad_bottom)`。
  - **data_format** (str，可选) - 指定输入的数据格式，可为 ``'NCHW'`` 或 ``'NHWC'``。默认值：``'NCHW'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，填充后的张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.CircularPad2D
