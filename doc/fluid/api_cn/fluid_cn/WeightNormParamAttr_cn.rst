.. _cn_api_fluid_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------

.. py:class:: paddle.fluid.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)


权重归一化。权重归一化是将权重向量的长度与其方向解耦。`Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_ 这篇paper中讨论了权重归一化的实现

参数:
  - **dim** (list) - 参数的名称。默认None。
  - **name** (str) - 参数的名称。默认None。
  - **initializer** （initializer) - 初始化参数的方法。默认None。
  - **learning_rate** (float) - 学习率。优化时学习速率 :math:`global\_lr∗parameter\_lr∗scheduler\_factor` 。默认1.0。
  - **regularizer** (WeightDecayRegularizer) - 正则化因子。默认None。
  - **trainable** (bool) - 参数是否可训练。默认True。
  - **gradient_clip** (BaseGradientClipAttr) - 梯度下降裁剪（Gradient Clipping）的方法。默认None。
  - **do_model_average** (bool) - 参数是否应该model average。默认False。

返回： empty program

**代码示例**

.. code-block:: python
      
  import paddle.fluid as fluid
  data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
  fc = fluid.layers.fc(input=data,
                       size=1000,
                       param_attr=fluid.WeightNormParamAttr(
                                dim=None,
                                name='weight_norm_param'))








