.. _cn_api_paddle_nn_AdaptiveAvgPool1D:

AdaptiveAvgPool1D
-------------------------------

.. py:class:: paddle.nn.AdaptiveAvgPool1D(output_size, name=None)

根据 ``output_size`` 对一个输入 Tensor 计算 1D 的自适应平均池化。输入和输出都是以 NCL 格式表示的 3-D Tensor，其中 N 是批大小，C 是通道数而 L 是特征的长度。输出的形状是 :math:`[N, C, output\_size]`。

计算公式为

..  math::

    lstart &= \lfloor i * L_{in} / L_{out}\rfloor,

    lend &= \lceil(i + 1) * L_{in} / L_{out}\rceil,

    Output(i) &= \frac{\sum Input[lstart:lend]}{lend - lstart}.


参数
:::::::::
    - **output_size** (int) - 输出特征的长度，数据类型为 int。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
用于计算 1D 自适应平均池化的可调用对象。


代码示例
:::::::::

COPY-FROM: paddle.nn.AdaptiveAvgPool1D
