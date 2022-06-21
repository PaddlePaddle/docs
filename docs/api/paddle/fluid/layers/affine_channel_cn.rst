.. _cn_api_fluid_layers_affine_channel:

affine_channel
-------------------------------

.. py:function:: paddle.fluid.layers.affine_channel(x, scale=None, bias=None, data_layout='NCHW', name=None,act=None)




对输入的每个 channel 应用单独的仿射变换。用于将空间批量归一化替换为其等价的固定变换。

输入也可以是二维张量，并在第二维应用仿射变换。

参数
::::::::::::

  - **x** (Variable)：特征图输入可以是一个具有NCHW格式或NHWC格式的的4-D张量。它也可以是二维张量，此时该算法应用于第二维度的仿射变换。数据类型为float32或float64。
  - **scale** (Variable)：维度为(C)的一维输入，第C个元素为输入的第C通道仿射变换的尺度因子。数据类型为float32或float64。
  - **bias** (Variable)：维度为(C)的一维输入，第C个元素是输入的第C个通道的仿射变换的偏置。数据类型为float32或float64。
  - **data_layout** (str，可选)：指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。如果输入是一个2D张量，可以忽略该参数，默认值为"NCHW"。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **act** (str，可选)：应用于该层输出的激活函数，默认值为None。

返回
::::::::::::
与x具有相同维度和数据布局的张量，数据类型与x相同

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.affine_channel