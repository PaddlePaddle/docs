.. _cn_api_nn_PReLU:

PReLU
-------------------------------
.. py:class:: paddle.nn.PReLU(num_parameters=1, init=0.25, weight_attr=None, data_format="NCHW", name=None)

PReLU 激活层（PReLU Activation Operator）。计算公式如下：

如果使用近似计算：

.. math::

    PReLU(x) = max(0, x) + weight * min(0, x)

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **num_parameters** (int，可选) - 可训练 ``weight``数量，支持的输入为：1 - 所有输入通道使用单一元素 ``alpha``；输入的通道数 - 在每个输入通道中使用独立的元素 ``alpha``。默认为 1。
    - **init** (float，可选) - 可学习的 ``weight``的初值。默认为 0.25。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (str，可选) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" 或者 "NDHWC"。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状：
::::::::::
    - input：任意形状的 Tensor，默认数据类型为 float32。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.PReLU

输出 (x)
:::::::::
    定义每次调用时执行的计算。应被所有子类覆盖。

参数
:::::::::
    - **inputs** (tuple) - 未压缩的 tuple 参数。
    - **kwargs** (dict) - 未压缩的字典参数。

extra_repr()
:::::::::
    该层为额外层，您可以自定义实现层。
