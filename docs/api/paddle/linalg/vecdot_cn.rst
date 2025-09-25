.. _cn_api_paddle_linalg_vecdot:

vecdot
-------------------------------

.. py:function:: paddle.linalg.vecdot(x, y, axis=-1, name=None)
计算两个张量沿指定轴的点积。

该函数对两个张量进行逐元素相乘，并沿指定轴求和以计算它们的点积。支持任意维度的张量（包括 0 维张量），只要张量`x`和`y`的形状在指定轴上可广播即可。


参数
:::::::::

        - **x** (Tensor) - 第一个输入张量。支持  ``float32``  、  ``float64``  、  ``int32``  、  ``int64``  、  ``complex64``  和   ``complex128``  六种数据类型。
        - **y** (Tensor) - 第二个输入张量。其形状必须能沿指定  ``axis``  与  ``x``  进行广播运算，且必须与  ``x``  具有相同的数据类型。
        - **axis** (int，可选) - 计算点积所沿的轴。默认为-1，表示最后一个轴。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
 ``Tensor`` ， 包含沿指定轴计算的  ``x``  和  ``y``  点积结果的张量。


代码示例
:::::::::
COPY-FROM: paddle.linalg.vecdot
