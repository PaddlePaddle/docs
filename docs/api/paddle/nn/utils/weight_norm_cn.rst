.. _cn_api_paddle_nn_utils_weight_norm:

weight_norm
-------------------------------

.. py:function:: paddle.nn.utils.weight_norm(layer, name='weight', dim=0)

根据以下公式对传入的 ``layer`` 中的权重参数进行归一化：

.. math::
    \mathbf{w} = g \dfrac{v}{\|v\|}

权重归一化可以将神经网络中权重向量的长度与其方向解耦，权重归一化可以用两个变量(例如：代表长度的变量 `weight_g` 和代表方向的变量 `weight_v`)来代替由名字(例如：`weight`)指定的变量。详细可以参考论文：`A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_

参数
::::::::::::

   - **layer** (paddle.nn.Layer) - 要添加权重归一化的层。
   - **name** (str，可选) - 权重参数的名字。默认值为 ``weight``。
   - **dim** (int|None，可选) - 进行归一化操作的切片所在维度，是小于权重 Tensor rank 的非负数。比如卷积的权重 shape 是 [cout,cin,kh,kw] , rank 是 4，则 dim 可以选 0,1,2,3；fc 的权重 shape 是 [cout,cin] ，rank 是 2，dim 可以选 0，1。如果为 None 就对所有维度上的元素做归一化。默认：0。

返回
::::::::::::

   ``Layer``，添加了权重归一化 hook 的层。

代码示例
::::::::::::

COPY-FROM: paddle.nn.utils.weight_norm
