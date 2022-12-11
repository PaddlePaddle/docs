.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------


.. py:class:: paddle.static.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, do_model_average=False, need_clip=True)


.. note::
    动态图模式下请使用 ``paddle.nn.utils.weight_norm`` 。

.. note::
    该类中的 ``gradient_clip`` 属性在 2.0 版本会废弃，推荐在初始化 ``optimizer`` 时设置梯度裁剪。共有三种裁剪策略：:ref:`cn_api_paddle_nn_ClipGradByGlobalNorm` 、
    :ref:`cn_api_paddle_nn_ClipGradByNorm` 、 :ref:`cn_api_paddle_nn_ClipGradByValue` 。

该类定义了权重归一化(Weight Normalization)的参数。权重归一化可以将神经网络中权重向量的长度与其方向解耦，详细的定义与实现可以参考论文：`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_

参数
::::::::::::

  - **dim** (int，可选) - 进行归一化操作(norm)的切片所在维度，是小于权重 Tensor rank 的非负数。比如卷积的权重 shape 是 :math:`[cout, cin, kh, kw]` , rank 是 4，则 dim 可以选 0,1,2,3；fc 的权重 shape 是 :math:`[cout, cin]` ，rank 是 2，dim 可以选 0，1。 dim 默认为 None，如果为 None 就对所有元素做归一化(norm)。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **initializer** （Initializer，可选) - 初始化参数方法，例如 ``initializer = fluid.nn.initializer.Constant(1.0)``。默认为 None，如果为 None 则使用默认初始化函数 `Xavier()` 。
  - **learning_rate** (float32，可选) - 学习率，优化过程 :math:`global\_lr∗parameter\_lr∗scheduler\_factor` 的学习速率，默认为 1.0。
  - **regularizer** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略：:ref:`cn_api_paddle_regularizer_L1Decay` 、
    :ref:`cn_api_paddle_regularizer_L2Decay`，如果在 ``optimizer`` (例如 :ref:`cn_api_paddle_optimizer_SGD` ) 中也
    设置了正则化，``optimizer`` 中的正则化将被忽略。默认值为 None，表示没有正则化。
  - **trainable** (bool) - 可选，指明参数是否可训练，默认为 True。
  - **do_model_average** (bool) - 可选，指明参数是否需要模型平均化操作(Model Average)，默认为 False。
  - **need_clip** (bool) - 可选，指明参数梯度是否需要在优化器中进行 clip，默认为 True。


代码示例
::::::::::::

COPY-FROM: paddle.static.WeightNormParamAttr
