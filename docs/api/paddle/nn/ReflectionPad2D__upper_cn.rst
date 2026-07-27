.. _cn_api_paddle_nn_ReflectionPad2D__upper:

ReflectionPad2D
-------------------------------
.. py:class:: paddle.nn.ReflectionPad2D(padding, data_format="NCHW", name=None)

使用输入边界的反射来填充输入张量的边界。

参数
::::::::::::

  - **padding** (Tensor|Sequence[int]|int) - 填充大小。如果 `padding` 是一个整数，则在所有四个边（左、右、上、下）应用相同的填充。如果 `padding` 是一个包含四个整数的列表或元组，它将被解析为 `(pad_left, pad_right, pad_top, pad_bottom)`。填充宽度必须小于相应输入的维度。
  - **data_format** (str，可选) - 指定输入的数据格式，可为 ``'NCHW'`` 或 ``'NHWC'``。默认值：``'NCHW'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，填充后的张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ReflectionPad2D
