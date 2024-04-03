.. _cn_api_paddle_nn_ZeroPad1D:

ZeroPad1D
-------------------------------
.. py:class:: paddle.nn.ZeroPad1D(padding, data_format="NCL", name=None)

**ZeroPad1D**

按照 padding 属性对输入进行零填充。

参数
:::::::::

  - **padding** (Tensor | List[int] | int) - 填充大小。如果是 int，则在所有待填充边界使用相同的填充，
    否则填充的格式为[pad_left, pad_right, pad_top, pad_bottom]。
  - **data_format** (str)  - 指定输入的 format，可为 ``'NCL'`` 或者 ``'NLC'``，默认值为 ``'NCL'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
无

形状
:::::::::

  - x(Tensor): ZeroPadD 层的输入，要求形状为 3-D，dtype 为 ``'float32'`` 或 ``'float64'``
  - output(Tensor)：输出，形状为 3-D，dtype 与 ``'input'`` 相同

代码示例
:::::::::

COPY-FROM: paddle.nn.ZeroPad1D
