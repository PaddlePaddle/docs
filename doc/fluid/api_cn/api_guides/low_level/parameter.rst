..  _api_guide_parameter:

#########
模型参数
#########

模型参数为模型中的weight和bias统称，在fluid中对应fluid.Parameter类，继承自fluid.Variable，是一种可持久化的variable。模型的训练就是不断学习更新模型参数的过程。模型参数相关的属性可以通过 :ref:`api_fluid_param_attr_ParamAttr` 来配置，可配置内容有：

- 初始化方式
- 正则化
- 梯度剪切
- 模型平均

初始化方式
=================

fluid通过设置 :code:`ParamAttr` 的 :code:`initializer` 属性为单个parameter设置初始化方式。
示例如下：

  .. code-block:: python

      param_attrs = fluid.ParamAttr(name="fc_weight",
                                initializer=fluid.initializer.ConstantInitializer(1.0))
      y_predict = fluid.layers.fc(input=x, size=10, param_attr=param_attrs)


以下为fluid支持的初始化方式：

1. BilinearInitializer
-----------------------

线性初始化方法。用该方法初始化的反卷积操作可当做线性插值操作使用。

可用别名：Bilinear

API请参考：:ref:`api_fluid_initializer_BilinearInitializer`

2. ConstantInitializer
----------------------

常数初始化方式，将parameter初始化为指定的数值。

可用别名：Constant

API请参考：:ref:`api_fluid_initializer_ConstantInitializer`

3. MSRAInitializer
------------------

该初始化方法参考论文: https://arxiv.org/abs/1502.01852

可用别名：MSRA

API请参考：:ref:`api_fluid_initializer_MSRAInitializer`

4. NormalInitializer
---------------------

随机高斯分布初始化方法。

可用别名：Normal

API请参考：:ref:`api_fluid_initializer_NormalInitializer`

5. TruncatedNormalInitializer
-----------------------------

随机截断高斯分布初始化方法。

可用别名：TruncatedNormal

API请参考：:ref:`api_fluid_initializer_TruncatedNormalInitializer`

6. UniformInitializer
--------------------

随机均匀分布初始化方式。

可用别名：Uniform

API请参考：:ref:`api_fluid_initializer_UniformInitializer`

7. XavierInitializer
--------------------

该初始化方式参考论文: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

可用别名：Xavier

API请参考：:ref:`api_fluid_initializer_XavierInitializer`

正则化方式
=============

fluid通过设置 :code:`ParamAttr` 的 :code:`regularizer` 属性为单个parameter设置正则化。

  .. code-block:: python

      param_attrs = fluid.ParamAttr(name="fc_weight",
                                regularizer=fluid.regularizer.L1DecayRegularizer(0.1))
      y_predict = fluid.layers.fc(input=x, size=10, param_attr=param_attrs)

以下为fluid支持的正则化方式：

- :ref:`api_fluid_regularizer_L1DecayRegularizer` (别名：L1Decay)
- :ref:`api_fluid_regularizer_L2DecayRegularizer` (别名：L2Decay)

Clipping
==========

fluid通过设置 :code:`ParamAttr` 的 :code:`gradient_clip` 属性为单个parameter设置clipping方式。

  .. code-block:: python

      param_attrs = fluid.ParamAttr(name="fc_weight",
                                regularizer=fluid.regularizer.L1DecayRegularizer(0.1))
      y_predict = fluid.layers.fc(input=x, size=10, param_attr=param_attrs)


以下为fluid支持的clipping方式：

1. ErrorClipByValue
-------------------

用来将一个tensor的值clipping到指定范围。

API请参考：:ref:`api_fluid_clip_ErrorClipByValue`

2. GradientClipByGlobalNorm
---------------------------

用来将多个Tensor的global-norm限制在 :code:`clip_norm` 以内。

API请参考：:ref:`api_fluid_clip_GradientClipByGlobalNorm`

3. GradientClipByNorm
---------------------

将Tensor的l2-norm限制在 :code:`max_norm` 以内。如果Tensor的l2-norm超过了 :code:`max_norm` ，
会将计算出一个 :code:`scale` ，该Tensor的所有值乘上计算出来的 :code:`scale` .

API请参考：:ref:`api_fluid_clip_GradientClipByNorm`

4. GradientClipByValue
----------------------

将parameter对应的gradient的值限制在[min, max]范围内。

API请参考：:ref:`api_fluid_clip_GradientClipByValue`

模型平均
========

fluid通过 :code:`ParamAttr` 的 :code:`do_model_average` 属性设置单个parameter是否进行平均优化。
示例如下：

  .. code-block:: python

      param_attrs = fluid.ParamAttr(name="fc_weight",
                                do_model_average=true)
      y_predict = fluid.layers.fc(input=x, size=10, param_attr=param_attrs)

在miniBatch训练过程中，每个batch过后，都会更新一次parameters，模型平均做的就是平均最近k次更新产生的parameters。

平均后的parameters只是被用来进行测试和预测，其并不参与实际的训练过程。

具体API请参考：:ref:`api_fluid_optimizer_ModelAverage`
