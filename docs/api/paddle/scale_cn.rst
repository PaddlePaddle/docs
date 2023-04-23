.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

对输入 Tensor 进行缩放和偏置，其公式如下：

``bias_after_scale`` 为 True:

.. math::
                        Out=scale*X+bias

``bias_after_scale`` 为 False:

.. math::
                        Out=scale*(X+bias)

参数
::::::::::::

        - **x** (Tensor) - 要进行缩放的多维 Tensor，数据类型可以为 float32，float64，int8，int16，int32，int64，uint8。
        - **scale** (float|Tensor) - 缩放的比例，是一个 float 类型或者一个 shape 为[1]，数据类型为 float32 的 Tensor 类型。
        - **bias** (float) - 缩放的偏置。
        - **bias_after_scale** (bool) - 判断在缩放之前或之后添加偏置。为 True 时，先缩放再偏置；为 False 时，先偏置再缩放。该参数在某些情况下，对数值稳定性很有用。
        - **act** (str，可选) - 应用于输出的激活函数，如 tanh、softmax、sigmoid、relu 等。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Tensor，缩放后的计算结果。

代码示例
::::::::::::

COPY-FROM: paddle.scale
