.. _cn_api_paddle_nn_ZeroPad3D:

ZeroPad3D
-------------------------------
.. py:class:: paddle.nn.ZeroPad3D(padding, data_format="NCDHW", name=None)

**ZeroPad3D**

按照 padding 属性对输入进行零填充。

参数
:::::::::

  - **padding** (Tensor | List[int] | int) - 如果输入数据类型为 int，则在所有待填充边界使用相同的填充，
    否则填充的格式为[pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]。
  - **data_format** (str，可选)  - 指定输入的 format，可为 ``'NCDHW'`` 或者 ``'NDHWC'``，默认值为 ``'NCDHW'``，其中 `N` 是批尺寸， `C` 是通道数， `D` 是特征层深度， `H` 是特征层高度， `W` 是特征层宽度。。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
无

形状
:::::::::

  - x(Tensor): ZeroPadD 层的输入，要求形状为 5-D，dtype 为 ``'float32'`` 或 ``'float64'``
  - output(Tensor)：输出，形状为 5-D，dtype 与 ``'input'`` 相同

代码示例
:::::::::

COPY-FROM: paddle.nn.ZeroPad3D
