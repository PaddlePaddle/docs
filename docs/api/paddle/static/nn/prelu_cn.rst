.. _cn_api_paddle_static_nn_prelu:

prelu
-------------------------------

.. py:function:: paddle.static.nn.prelu(x, mode, param_attr=None, data_format="NCHW", name=None)

prelu 激活函数

.. math::
    prelu(x) = max(0, x) + \alpha * min(0, x)

共提供三种激活方式：

    - all：所有元素使用同一个 alpha 值；
    - channel：在同一个通道中的元素使用同一个 alpha 值；
    - element：每一个元素有一个独立的 alpha 值。


参数
::::::::::::

    - **x** （Tensor）- 多维 Tensor，数据类型为 float32。
    - **mode** (str) - 权重共享模式。
    - **param_attr** (ParamAttr，可选) - 可学习权重 :math:`[\alpha]` 的参数属性，可由 ParamAttr 创建。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **data_format** (str，可选) – 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" 或者 "NDHWC"。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
表示激活输出 Tensor，数据类型和形状于输入相同。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.prelu
