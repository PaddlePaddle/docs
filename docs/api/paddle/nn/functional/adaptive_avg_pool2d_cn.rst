.. _cn_api_nn_functional_adaptive_avg_pool2d:

adaptive_avg_pool2d
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_avg_pool2d(x, output_size, data_format='NCHW', name=None)

根据输入 `x` , `output_size` 等参数对一个输入 Tensor 计算 2D 的自适应平均池化。输入和输出都是 4-D Tensor，
默认是以 `NCHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`H` 是输入特征的高度，`H` 是输入特征的宽度。

计算公式如下：

..  math::

    hstart &= floor(i * H_{in} / H_{out})

    hend &= ceil((i + 1) * H_{in} / H_{out})

    wstart &= floor(j * W_{in} / W_{out})

    wend &= ceil((j + 1) * W_{in} / W_{out})

    Output(i ,j) &= \frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}


参数
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，高度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 float16, float32, float64, int32 或 int64。
    - **output_size** (int|list|turple)：算子输出特征图的尺寸，如果其是 list 或 turple 类型的数值，必须包含两个元素，H 和 W。H 和 W 既可以是 int 类型值也可以是 None，None 表示与输入特征尺寸相同。
    - **data_format** (str)：输入和输出的数据格式，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，默认形状为（批大小，通道数，输出特征高度，输出特征宽度），即 NCHW 格式的 4-D Tensor，其数据类型与输入相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.adaptive_avg_pool2d
