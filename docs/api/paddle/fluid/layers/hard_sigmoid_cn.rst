.. _cn_api_fluid_layers_hard_sigmoid:

hard_sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)




sigmoid 的分段线性逼近激活函数，速度比 sigmoid 快，详细解释参见 https://arxiv.org/abs/1603.00391。

.. math::

      \\out=\max(0,\min(1,slope∗x+offset))\\

参数
::::::::::::

    - **x** (Variable) - 该 OP 的输入为多维 Tensor。数据类型必须为 float32 或 float64。
    - **slope** (float，可选) - 斜率。值必须为正数，默认值为 0.2。
    - **offset** (float，可选) - 偏移量。默认值为 0.5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
激活后的 Tensor，形状、数据类型和 ``x`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.hard_sigmoid
