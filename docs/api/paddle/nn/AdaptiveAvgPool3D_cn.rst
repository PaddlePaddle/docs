.. _cn_api_nn_AdaptiveAvgPool3D:

AdaptiveAvgPool3D
-------------------------------

.. py:function:: paddle.nn.AdaptiveAvgPool3D(output_size, data_format="NCDHW", name=None)

根据输入 `x` , `output_size` 等参数对一个输入 Tensor 计算 3D 的自适应平均池化。输入和输出都是 5-D Tensor，
默认是以 `NCDHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`D` 是特征图长度，`H` 是输入特征的高度，`W` 是输入特征的宽度。

计算公式如下：

..  math::

    dstart &= floor(i * D_{in} / D_{out})

    dend &= ceil((i + 1) * D_{in} / D_{out})

    hstart &= floor(j * H_{in} / H_{out})

    hend &= ceil((j + 1) * H_{in} / H_{out})

    wstart &= floor(k * W_{in} / W_{out})

    wend &= ceil((k + 1) * W_{in} / W_{out})

    Output(i ,j, k) &= \frac{\sum Input[dstart:dend, hstart:hend, wstart:wend]}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

参数
:::::::::
    - **output_size** (int|list|tuple)：算子输出特征图的尺寸，如果其是 list 或 turple 类型的数值，必须包含三个元素，D，H 和 W。D，H 和 W 既可以是 int 类型值也可以是 None，None 表示与输入特征尺寸相同。
    - **data_format** (str，可选)：输入和输出的数据格式，可以是"NCDHW"和"NDHWC"。N 是批尺寸，C 是通道数，D 是特征长度，H 是特征高度，W 是特征宽度。默认值："NCDHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，长度，高度，宽度），即 NCDHW 格式的 5-D Tensor。其数据类型为 float16, float32, float64, int32 或 int64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度，输出特征高度，输出特征宽度），即 NCDHW 格式的 5-D Tensor。其数据类型与输入相同。


返回
:::::::::
计算 AdaptiveAvgPool3D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.AdaptiveAvgPool3D
