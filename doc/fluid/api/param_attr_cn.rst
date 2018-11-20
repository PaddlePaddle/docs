

.. _cn_api_fluid_param_attr_ParamAttr:

ParamAttr
>>>>>>>>>>>>

.. py:class:: paddle.fluid.param_attr.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

参数属性对象。为了对网络训练过程进行微调，用户可以设置参数属性来控制训练细节。如学习率、正则化、可训练、do_model_average和初始化参数的方法


参数:
  - **name**  (str) -参数的名称 默认为None
  - **initializer** (initializer) - 初始化这个参数的方法 默认None。
  - **learning_rate**  (float) - 参数的学习率 优化时学习速率为 global_lr∗parameter_lr∗scheduler_factor 默认 1.0
  - **regularizer** （WeightDecayRegularizer）- 正则化因子。默认None
  - **可训练** (bool) - 这个参数是否可训练。默认 True
  - **gradient_clip**  (BaseGradientClipAttr) - 修剪这个参数的梯度的方法 默认 None
  - **do_model_average**  (bool) - 这个参数是否应该做模型平均 默认 False。

**代码示例**


..  code-block:: python
  
    w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                learning_rate=0.5,
                                regularizer=fluid.L2Decay(1.0),
                                trainable=True)
    y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)
