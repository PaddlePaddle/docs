.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------

.. py:class:: paddle.fluid.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)


该类定义了权重归一化(Weight Normalization)的参数。权重归一化可以将神经网络中权重向量的长度与其方向解耦，详细的定义与实现可以参考论文：`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_

参数:
  - **dim** (int) - 进行归一化操作(norm)的切片所在维度，是小于用户权重rank的非负数。比如卷积的权重shape是 :math:`[cout, cin, kh, kw]` , rank是4，则dim可以选0,1,2,3; fc的权重shape是 :math:`[cout cin]` ，rank是2，dim可以选0，1。 dim 默认为None，如果为None就对所有元素做归一化(norm)，不做切片。
  - **name** (None|str) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认为None。
  - **initializer** （Initializer) - 初始化参数方法类，例如 ``initializer = fluid.initializer.ConstantInitializer(1.0)`` ，Paddle中 ``Initializer`` 类的详细定义参见 :ref:`cn_api_fluid_initializer_Bilinear` 。默认为None，如果为None则使用默认初始化函数 `Xavier()` 。
  - **learning_rate** (float32) - 学习率，优化过程 :math:`global\_lr∗parameter\_lr∗scheduler\_factor` 的学习速率，默认为1.0。
  - **regularizer** (WeightDecayRegularizer) - 正则化方法类，例如 ``regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.1)`` 。Paddle中 ``regularizer`` 类的详细定义参见 :ref:`cn_api_fluid_regularizer_L1DecayRegularizer` 。默认为None，如果为None就不做正则化。
  - **trainable** (bool) - 可选，指明参数是否可训练，默认为True。
  - **gradient_clip** - 梯度裁剪(Gradient Clipping)的方法类，例如 ``gradient_clip = fluid.clip.GradientClipByNorm(clip_norm=2.0))`` 。Paddle中 ``GradientClip`` 类的详细定义参见 :ref:`cn_api_fluid_clip_GradientClipByNorm` 。默认为None，如果为None就不做裁剪。
  - **do_model_average** (bool) - 可选，指明参数是否需要模型平均化操作(Model Average)，默认为False。


**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
  fc = fluid.layers.fc(input=data,
                       size=1000,
                       param_attr=fluid.WeightNormParamAttr(
                                dim=None,
                                name='weight_norm_param',
                                initializer=fluid.initializer.ConstantInitializer(1.0),
                                learning_rate=1.0,
                                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.1),
                                trainable=True,
                                gradient_clip=fluid.clip.GradientClipByNorm(clip_norm=2.0),
                                do_model_average=False))



