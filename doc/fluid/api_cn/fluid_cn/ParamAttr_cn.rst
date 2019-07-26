.. _cn_api_fluid_ParamAttr:


ParamAttr
-------------------------------


.. py:class:: paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

该类代表了参数的各种属性。 为了使神经网络训练过程更加流畅，用户可以根据需要调整参数属性。比如learning rate（学习率）, regularization（正则化）, trainable（可训练性）, do_model_average(平均化模型)和参数初始化方法.

参数:
    - **name** (str) – 参数名。默认为None。
    - **initializer** (Initializer) – 初始化该参数的方法。 默认为None
    - **learning_rate** (float) – 参数的学习率。计算方法为 :math:`global\_lr*parameter\_lr∗scheduler\_factor` 。 默认为1.0
    - **regularizer** (WeightDecayRegularizer) – 正则因子. 默认为None
    - **trainable** (bool) – 该参数是否可训练。默认为True
    - **gradient_clip** (BaseGradientClipAttr) – 减少参数梯度的方法。默认为None
    - **do_model_average** (bool) – 该参数是否服从模型平均值。默认为False

**代码示例**

.. code-block:: python

   import paddle.fluid as fluid
   
   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.L2Decay(1.0),
                                   trainable=True)
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)













