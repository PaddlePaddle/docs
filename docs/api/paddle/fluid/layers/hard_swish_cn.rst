.. _cn_api_fluid_layers_hard_swish:

hard_swish
-------------------------------

.. py:function:: paddle.fluid.layers.hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None)




该OP实现了hard_swish激活函数。hard_swish激活函数在MobileNetV3架构中被提出，相较于swish函数，具有数值稳定性好，计算速度快等优点，具体原理请参考：https://arxiv.org/pdf/1905.02244.pdf

 :math:`out = \frac{x * (min(max(0, x+offset), threshold))}{scale}`

 阈值 ``threshold`` 和缩放因子 ``scale`` 为正数，位移 ``offset`` 正负均可，建议使用默认参数。

参数
::::::::::::

    - **x** (Variable) - 输入特征，多维Tensor。数据类型为float32或float64。
    - **threshold** (float，可选) - 激活操作中Relu函数的阈值，默认值为6.0。 
    - **scale** (float，可选) - 激活操作的缩放因子，默认值为6.0。
    - **offset** (float，可选) - 激活操作的位移，默认值为3.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    
返回
::::::::::::
经过hard_swish计算后的结果，数据类型及维度和x相同。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.hard_swish