.. _cn_api_nn_AdaptiveAvgPool2D:

AdaptiveAvgPool2D
-------------------------------

.. py:function:: paddle.nn.AdaptiveAvgPool2D(output_size, data_format="NCHW", name=None)

该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算2D的自适应平均池化。输入和输出都是4-D Tensor，
默认是以 `NCHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`H` 是输入特征的高度，`W` 是输入特征的宽度。

计算公式如下：

..  math::

    hstart &= floor(i * H_{in} / H_{out})

    hend &= ceil((i + 1) * H_{in} / H_{out})

    wstart &= floor(j * W_{in} / W_{out})

    wend &= ceil((j + 1) * W_{in} / W_{out})

    Output(i ,j) &= \frac{\sum Input[hstart:hend, wstart:wend]}{(hend - hstart) * (wend - wstart)}


参数
:::::::::
    - **output_size** (int|list|tuple)：算子输出特征图的尺寸。如果其是list或turple类型的数值，必须包含两个元素：H和W。H和W既可以是int类型值也可以是None，None表示与输入特征尺寸相同。
    - **data_format** (str，可选)：输入和输出的数据格式，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，高度，宽度），即NCHW格式的4-D Tensor。其数据类型为float16, float32, float64, int32或int64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即NCHW格式的4-D Tensor。其数据类型与输入相同。


返回
:::::::::
计算AdaptiveAvgPool2D的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.AdaptiveAvgPool2D