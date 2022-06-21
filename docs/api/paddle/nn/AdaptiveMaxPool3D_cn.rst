.. _cn_api_nn_AdaptiveMaxPool3D:


AdaptiveMaxPool3D
-------------------------------

.. py:function:: paddle.nn.AdaptiveMaxPool3D(output_size, return_mask=False, name=None)
该算子根据输入 `x` , `output_size` 等参数对一个输入Tensor计算3D的自适应最大池化。输入和输出都是5-D Tensor，
默认是以 `NCDHW` 格式表示的，其中 `N` 是 batch size， `C` 是通道数，`D` ， `H` ， `W` 分别是输入特征的深度，高度，宽度。

计算公式如下：

..  math::

    dstart &= floor(i * D_{in} / D_{out})
    
    dend &= ceil((i + 1) * D_{in} / D_{out})
    
    hstart &= floor(j * H_{in} / H_{out})
    
    hend &= ceil((j + 1) * H_{in} / H_{out})
    
    wstart &= floor(k * W_{in} / W_{out})
    
    wend &= ceil((k + 1) * W_{in} / W_{out})
    
    Output(i ,j, k) &= max(Input[dstart:dend, hstart:hend, wstart:wend])

参数
:::::::::
    - **output_size** (int|list|tuple)：算子输出特征图的高宽长大小，其数据类型为int,list或tuple。
    - **return_mask** (bool，可选)：如果设置为True，则会与输出一起返回最大值的索引，默认为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **x** (Tensor)：默认形状为（批大小，通道数，输出特征深度，高度，宽度），即NCDHW格式的5-D Tensor。其数据类型为float32或者float64。
    - **output** (Tensor)：默认形状为（批大小，通道数，输出特征深度，高度，宽度），即NCDHW格式的5-D Tensor。其数据类型与输入x相同。

返回
:::::::::
计算AdaptiveMaxPool3D的可调用对象


代码示例
:::::::::

COPY-FROM: paddle.nn.AdaptiveMaxPool3D