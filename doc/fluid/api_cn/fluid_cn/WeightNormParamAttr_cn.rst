.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------


.. py:class:: paddle.fluid.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, do_model_average=False)

:api_attr: 声明式编程模式（静态图)



.. note::
    该类中的 ``gradient_clip`` 属性在2.0版本会废弃，推荐在初始化 ``optimizer`` 时设置梯度裁剪。共有三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 
    :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。

该类定义了权重归一化(Weight Normalization)的参数。权重归一化可以将神经网络中权重向量的长度与其方向解耦，详细的定义与实现可以参考论文：`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_

参数:
  - **dim** (int) - 进行归一化操作(norm)的切片所在维度，是小于权重Tensor rank的非负数。比如卷积的权重shape是 :math:`[cout, cin, kh, kw]` , rank是4，则dim可以选0,1,2,3；fc的权重shape是 :math:`[cout, cin]` ，rank是2，dim可以选0，1。 dim 默认为None，如果为None就对所有元素做归一化(norm)。
  - **name** (None|str) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认为None。
  - **initializer** （Initializer) - 初始化参数方法，例如 ``initializer = fluid.initializer.ConstantInitializer(1.0)`` 。默认为None，如果为None则使用默认初始化函数 `Xavier()` 。
  - **learning_rate** (float32) - 学习率，优化过程 :math:`global\_lr∗parameter\_lr∗scheduler\_factor` 的学习速率，默认为1.0。
  - **regularizer** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
    :ref:`cn_api_fluid_regularizer_L2Decay` ，如果在 ``optimizer`` (例如 :ref:`cn_api_fluid_optimizer_SGDOptimizer` ) 中也
    设置了正则化，``optimizer`` 中的正则化将被忽略。默认值为None，表示没有正则化。
  - **trainable** (bool) - 可选，指明参数是否可训练，默认为True。
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
                                do_model_average=False))



