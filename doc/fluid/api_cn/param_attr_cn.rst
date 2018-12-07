#################
 fluid.param_attr
#################




.. _cn_api_fluid_param_attr_ParamAttr:

 
ParamAttr
-------------------------------


.. py:class:: paddle.fluid.param_attr.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

该类代表了参数的各种属性。 为了使神经网络训练过程更加流畅，用户可以根据需要调整参数属性。比如learning rate（学习率）, regularization（正则化）, trainable（可训练性）, do_model_average(平均化模型)和参数初始化方法.

参数:	
    - **name** (str) – 参数名。默认为None。
    - **initializer** (Initializer) – 初始化该参数的方法。 默认为None
    - **learning_rate** (float) – 参数的学习率。计算方法为 global_lr*parameter_lr∗scheduler_factor。 默认为1.0
    - **regularizer** (WeightDecayRegularizer) – 正则因子. 默认为None
    - **trainable** (bool) – 该参数是否可训练。默认为True
    - **gradient_clip** (BaseGradientClipAttr) – 减少参数梯度的方法。默认为None
    - **do_model_average** (bool) – 该参数是否服从模型平均值。默认为False
    
**代码示例**

..  code-block:: python

   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.L2Decay(1.0),
                                   trainable=True)
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)



英文版API文档: :ref:`api_fluid_param_attr_ParamAttr` 






.. _cn_api_fluid_param_attr_WeightNormParamAttr:

WeightNormParamAttr
-------------------------------

.. py:class:: paddle.fluid.param_attr.WeightNormParamAttr(dim=None, name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)
  
权重归一化。权范数是神经网络中权向量的再参数化，它将权向量的长度与其方向解耦。该paper对权值归一化的实现进行了讨论： `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_ 

参数:
  - **dim**  (list) – 参数维度. Default None.
  - **name** (str) – 参数名称. Default None.
  - **initializer**  (Initializer) – 初始化参数的方法. Default None.
  - **learning_rate**  (float) – 参数的学习率. 优化的参数学习率为 :math:`global\_lr*parameter\_lr*scheduler\_factor` . Default 1.0
  - **regularizer**  (WeightDecayRegularizer) – 正则化因子. Default None.
  - **trainable**  (bool) – 参数是否可训练. Default True.
  - **gradient_clip**  (BaseGradientClipAttr) – 修剪这个参数的梯度的方法. Default None.
  - **do_model_average**  (bool) – 这个参数是否应该做模型平均. Default False.


**代码示例**


..  code-block:: python
  
    data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
    fc = fluid.layers.fc(input=data,
                          size=1000,
                          param_attr=WeightNormParamAttr(
                          dim=None,
                          name='weight_norm_param'))
                          
             



英文版API文档: :ref:`api_fluid_param_attr_WeightNormParamAttr` 







