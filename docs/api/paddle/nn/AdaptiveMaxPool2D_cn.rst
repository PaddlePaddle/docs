.. _cn_api_nn_AdaptiveMaxPool2D:


AdaptiveMaxPool2D
-------------------------------

.. py:class:: paddle.nn.AdaptiveMaxPool2D(output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入 Tensor 计算 2D 的自适应最大池化。输入和输出都是 4-D Tensor，
默认是以 `NCHW` 格式表示的，其中 `N` 是 batch size, `C` 是通道数，`H` 是输入特征的高度，`W` 是输入特征的宽度。

计算公式如下：

..  math::

    lstart &= floor(i * L_{in} / L_{out})

    lend &= ceil((i + 1) * L_{in} / L_{out})

    Output(i) &= max(Input[lstart:lend])

    hstart &= floor(i * H_{in} / H_{out})

    hend &= ceil((i + 1) * H_{in} / H_{out})

    wstart &= floor(j * W_{in} / W_{out})

    wend &= ceil((j + 1) * W_{in} / W_{out})

    Output(i ,j) &= max(Input[hstart:hend, wstart:wend])

参数
:::::::::

    - **output_size** (int|list|tuple)：算子输出特征图的高和宽大小，其数据类型为 int,list 或 tuple。
    - **return_mask** (bool，可选)：如果设置为 True，则会与输出一起返回最大值的索引，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::

    - **x** (Tensor)：默认形状为（批大小，通道数，输出特征长度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型为 float32 或者 float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征长度，宽度），即 NCHW 格式的 4-D Tensor。其数据类型与输入 x 相同。

返回
:::::::::

计算 AdaptiveMaxPool2D 的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.AdaptiveMaxPool2D
