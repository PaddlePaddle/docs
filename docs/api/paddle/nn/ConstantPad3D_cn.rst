.. _cn_api_paddle_nn_ConstantPad3d:

ConstantPad3d
-------------------------------
.. py:class:: paddle.nn.ConstantPad3d(padding, value, data_format="NCDHW", name=None)

使用一个常量值来填充输入张量的边界。

参数
::::::::::::

  - **padding** (Tensor|Sequence[int]|int) - 填充大小。如果 `padding` 是一个整数，则在所有六个边（左、右、上、下、前、后）应用相同的填充。如果 `padding` 是一个包含六个整数的列表或元组，它将被解析为 `(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)`。
  - **value** (float) - 用于填充区域的常量值。
  - **data_format** (str，可选) - 指定输入的数据格式，可为 ``'NCDHW'`` 或 ``'NDHWC'``。默认值：``'NCDHW'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，填充后的张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ConstantPad3d
