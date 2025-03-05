..  _api_guide_parameter:

#########
模型参数
#########

.. note::
  `paddle.fluid.*` 已废弃，请使用 Paddle 最新版本的 API。

模型参数通常为模型中的 weight 和 bias，在 Paddle 动态图下对应 :code:`EagerParamBase` 类。模型参数是可学习的变量，拥有梯度并且可优化。在 Paddle 中可以通过 :ref:`cn_api_paddle_create_parameter` 来创建自定义参数。

模型参数相关的属性可以通过 :ref:`cn_api_paddle_ParamAttr` 来配置，可配置内容有：

- 初始化方式
- 正则化
- 模型平均
- 梯度剪切

初始化方式
=================

Paddle 通过设置 :code:`ParamAttr` 的 :code:`initializer` 属性为单个 parameter 设置初始化方式。
示例如下：

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                initializer=paddle.nn.initializer.Constant(5.0))
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

以下为 Paddle 支持的初始化方式：

1. Constant
------------

常量初始化方法。用于将参数初始化为指定的固定数值，如将偏置初始化为 0。

参数 `value` 指定初始值，默认值为 0.0。

API 请参考：:ref:`cn_api_paddle_nn_initializer_Constant`

2. Normal
----------

随机正态分布初始化方法。根据正态（高斯）分布生成随机值，适用于大多数神经网络的参数初始化。

参数指定均值 `mean` （默认 0.0）和标准差 `std` （默认 1.0）。

API 请参考：:ref:`cn_api_paddle_nn_initializer_Normal`

3. Uniform
-----------

随机均匀分布初始化方法。在指定区间 [low, high] 内均匀采样生成初始值。

参数 `low` 和 `high` 默认为 -1.0 和 1.0。

API 请参考：:ref:`cn_api_paddle_nn_initializer_Uniform`

4. XavierUniform
-----------------

Xavier 均匀分布初始化方法。源自论文 **Understanding the difficulty of training deep feedforward neural networks**，由 Xavier Glorot 和 Yoshua Bengio 提出。

范围由 `fan_in` （输入维度）、 `fan_out` （输出维度）和 `gain` （增益值）决定。

API 请参考：:ref:`cn_api_paddle_nn_initializer_XavierUniform`

5. XavierNormal
----------------

Xavier 正态分布初始化方法。源自论文 **Understanding the difficulty of training deep feedforward neural networks**。

均值为 0，标准差由 `fan_in` （输入维度）、 `fan_out` （输出维度） 和 `gain` （增益值） 决定。

API 请参考：:ref:`cn_api_paddle_nn_initializer_XavierNormal`

6. KaimingUniform
------------------

Kaiming 均匀分布初始化方法。源自论文 **Delving Deep into Rectifiers**，由 Kaiming He 等提出。

范围由 `fan_in` （输入维度）、 `negative_slope` （负斜率，默认 0）和 `nonlinearity` （激活类型，默认 'relu'）决定。

API 请参考：:ref:`cn_api_paddle_nn_initializer_KaimingUniform`

7. KaimingNormal
-----------------

Kaiming 正态分布初始化方法。源自论文 **Delving Deep into Rectifiers**。

均值为 0，标准差由 `fan_in` （输入维度）、 `negative_slope` （负斜率，默认 0）和 `nonlinearity` （激活类型，默认 'relu'）决定。

API 请参考：:ref:`cn_api_paddle_nn_initializer_KaimingNormal`

8. TruncatedNormal
-------------------

截断正态分布初始化方法。在正态分布基础上，限制生成值在指定范围 [a, b] 内。

参数指定均值 `mean` （默认 0.0）、标准差 `std` （默认 1.0）以及截断边界 `a` 和 `b` （默认 -2.0 和 2.0）。

API 请参考：:ref:`cn_api_paddle_nn_initializer_TruncatedNormal`

其他初始化方式
-----------------

Paddle 还支持以下初始化方式：

- :ref:`cn_api_paddle_nn_initializer_Assign`：通过 Numpy 数组、Python 列表或 Tensor 直接赋值初始化。
- :ref:`cn_api_paddle_nn_initializer_Bilinear`：用于转置卷积的上采样初始化，支持特征图放大。
- :ref:`cn_api_paddle_nn_initializer_Dirac`：通过 Dirac delta 函数初始化卷积核，保留输入特性。
- :ref:`cn_api_paddle_nn_initializer_Orthogonal`：生成正交矩阵初始化参数，被初始化的参数为 (半)正交的。
- :ref:`cn_api_paddle_nn_initializer_set_global_initializer`：设置全局参数初始化方法。只对位于其后的代码生效。

正则化方式
=============

Paddle 通过设置 :code:`ParamAttr` 的 :code:`regularizer` 属性为单个 parameter 设置正则化。

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                regularizer=paddle.regularizer.L1Decay(0.1))
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)


以下为 Paddle 支持的正则化方式：

- :ref:`cn_api_paddle_regularizer_L1Decay`
- :ref:`cn_api_paddle_regularizer_L2Decay`

模型平均
========

Paddle 通过设置 :code:`ParamAttr` 的 :code:`do_model_average` 属性为单个 parameter 设置是否进行平均优化。

默认值为 `True` 。

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                do_model_average=True)
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

在 mini-batch 训练过程中，每个 batch 过后，模型的 parameters 都会被更新一次。模型平均的作用就是平均最近 k 次更新产生的 parameters。

平均后的 parameters 仅用于测试和预测，不参与实际的训练过程。

API 请参考：:ref:`cn_api_paddle_incubate_ModelAverage` （当前处于孵化状态，API 可能会有变动）

Clipping
==========

.. note::
  :code:`gradient_clip` 已废弃，请使用 :code:`need_clip` 设置是否进行梯度裁剪，并在初始化 :code:`optimizer` 时设置梯度裁剪方法。

Paddle 通过设置 :code:`ParamAttr` 的 :code:`need_clip` 属性为单个 parameter 设置是否进行梯度裁剪。

默认值为 `True` 。

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                need_clip=True)
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

以下为 Paddle 支持的 clipping 方式：

1. GradientClipByGlobalNorm
---------------------------

将一个 Tensor 列表 `t_list` 中所有 Tensor 的 L2 范数之和，限定在 :code:`clip_norm` 范围内。

API 请参考：:ref:`cn_api_paddle_nn_ClipGradByGlobalNorm`

2. GradientClipByNorm
---------------------

将输入的多维 Tensor `X` 的 L2 范数限制在 :code:`clip_norm` 范围之内。

API 请参考：:ref:`cn_api_paddle_nn_ClipGradByNorm`

3. GradientClipByValue
----------------------

将输入的多维 Tensor `X` 的值限制在 [min, max] 范围之内。

API 请参考：:ref:`cn_api_paddle_nn_ClipGradByValue`

具体梯度裁剪方式请参考：:ref:`梯度裁剪方式介绍 <cn_gradient_clip>`
