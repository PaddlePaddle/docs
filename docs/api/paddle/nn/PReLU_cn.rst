.. _cn_api_paddle_nn_PReLU:

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
    - **num_parameters** (int，可选) - 可训练`weight`数量，支持 2 种输入：1 - 输入中的所有元素使用同一个`weight`值；输入的通道数 - 在同一个通道中的元素使用同一个`weight`值。默认为 1。
    - **init** (float，可选) - `weight`的初始值。默认为 0.25。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`_cn_api_paddle_ParamAttr` 。
    - **data_format** (str，可选) – 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是 "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" 或者 "NDHWC"。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor，默认数据类型为 float32。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.PReLU
