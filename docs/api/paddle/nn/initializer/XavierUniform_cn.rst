.. _cn_api_paddle_nn_initializer_XavierUniform:

XavierUniform
-------------------------------

.. py:class:: paddle.nn.initializer.XavierUniform(fan_in=None, fan_out=None, gain=1.0, name=None)


使用均匀分布的泽维尔权重初始化方法。泽维尔权重初始化方法出自泽维尔·格洛特和约书亚·本吉奥的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ 。

该初始化函数用于保持所有层的梯度尺度几乎一致。在均匀分布的情况下，取值范围为 :math:`[-x,x]`，其中

.. math::

    x = gain \times \sqrt{\frac{6.0}{fan\_in+fan\_out}}.

参数
::::::::::::

    - **fan_in** (float，可选) - 用于泽维尔初始化的 fan_in，从 Tensor 中推断，默认值为 None。
    - **fan_out** (float，可选) - 用于泽维尔初始化的 fan_out，从 Tensor 中推断，默认值为 None。
    - **gain** (float，可选) - 缩放因子。默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由使用均匀分布的泽维尔权重初始化方法得到的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.XavierUniform
