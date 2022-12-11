.. _cn_api_nn_initializer_KaimingUniform:

KaimingUniform
-------------------------------

.. py:class:: paddle.nn.initializer.KaimingUniform(fan_in=None, negative_slope=0.0, nonlinearity='relu')




Kaiming 均匀分布方式的权重初始化函数，方法来自 Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun 所写的论文：`Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 。这是一个鲁棒性特别强的初始化方法，并且适应了非线性激活函数（rectifier nonlinearities）。

在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = gain \times \sqrt{\frac{3}{fan\_in}}

参数
::::::::::::

    - **fan_in** (float16|float32，可选) - 可训练的 Tensor 的 in_features 值。如果设置为 None，程序会自动计算该值。如果你不想使用 in_features，你可以自己设置这个值。默认值为 None。
    - **negative_slope** (float，可选) -  只适用于使用 leaky_relu 作为激活函数时的 negative_slope 参数。默认值为 :math:`0.0`。
    - **nonlinearity** (str，可选) -  非线性激活函数。默认值为 relu。

.. note::

    在大多数情况下推荐设置 fan_in 为 None。

返回
::::::::::::
对象。



代码示例
::::::::::::
COPY-FROM: paddle.nn.initializer.KaimingUniform
