.. _cn_api_fluid_layers_affine_channel:

affine_channel
-------------------------------

.. py:function:: paddle.fluid.layers.affine_channel(x, scale=None, bias=None, data_layout='NCHW', name=None,act=None)




对输入的每个 channel 应用单独的仿射变换。用于将空间批量归一化替换为其等价的固定变换。

输入也可以是二维 Tensor，并在第二维应用仿射变换。

参数
::::::::::::

  - **x** (Variable)：特征图输入可以是一个具有 NCHW 格式或 NHWC 格式的的 4-DTensor。它也可以是二维 Tensor，此时该算法应用于第二维度的仿射变换。数据类型为 float32 或 float64。
  - **scale** (Variable)：维度为(C)的一维输入，第 C 个元素为输入的第 C 通道仿射变换的尺度因子。数据类型为 float32 或 float64。
  - **bias** (Variable)：维度为(C)的一维输入，第 C 个元素是输入的第 C 个通道的仿射变换的偏置。数据类型为 float32 或 float64。
  - **data_layout** (str，可选)：指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。如果输入是一个 2DTensor，可以忽略该参数，默认值为"NCHW"。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **act** (str，可选)：应用于该层输出的激活函数，默认值为 None。

返回
::::::::::::
与 x 具有相同维度和数据布局的 Tensor，数据类型与 x 相同

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.affine_channel
