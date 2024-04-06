.. _cn_api_paddle_nn_initializer_XavierNormal:

XavierNormal
-------------------------------

.. py:class:: paddle.nn.initializer.XavierNormal(fan_in=None, fan_out=None, gain=1.0, name=None)


使用正态分布的泽维尔权重初始化方法。泽维尔权重初始化方法出自泽维尔·格洛特和约书亚·本吉奥的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ 。

该初始化函数用于保持所有层的梯度尺度几乎一致。所使用的正态分布的的均值为 :math:`0`，标准差为

.. math::

    x = gain \times \sqrt{\frac{2.0}{fan\_in+fan\_out}}.

参数
::::::::::::

    - **fan_in** (float，可选) - 用于泽维尔初始化的 fan_in，从 Tensor 中推断，默认值为 None。
    - **fan_out** (float，可选) - 用于泽维尔初始化的 fan_out，从 Tensor 中推断，默认值为 None。
    - **gain** (float，可选) - 缩放因子。默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由使用正态分布的泽维尔权重初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.XavierNormal
