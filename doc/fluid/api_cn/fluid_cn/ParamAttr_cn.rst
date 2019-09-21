.. _cn_api_fluid_ParamAttr:


ParamAttr
-------------------------------


.. py:class:: paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, gradient_clip=None, do_model_average=False)

创建一个参数属性对象，用户可设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性。

参数:
    - **name** (str，可选) - 参数的名称。默认值为None，表示框架自动创建参数的名称。
    - **initializer** (Initializer，可选) - 参数的初始化方式。默认值为None，表示权重参数采用Xavier初始化方式，偏置参数采用全0初始化方式。
    - **learning_rate** (float) - 参数的学习率。实际参数的学习率等于全局学习率乘以参数的学习率，再乘以learning rate schedule的系数。
    - **regularizer** (WeightDecayRegularizer，可选) - 正则化因子。默认值为None，表示没有正则化因子。
    - **trainable** (bool) - 参数是否需要训练。默认值为True，表示需要训练。
    - **gradient_clip** (BaseGradientClipAttr，可选) - 梯度裁剪方式。默认值为None，表示不需要梯度裁剪。
    - **do_model_average** (bool) - 是否做模型平均。默认值为False，表示不做模型平均。

返回: 表示参数属性的对象。

返回类型: ParamAttr

**代码示例**

.. code-block:: python

   import paddle.fluid as fluid
   
   w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                   learning_rate=0.5,
                                   regularizer=fluid.regularizer.L2Decay(1.0),
                                   trainable=True)
   x = fluid.layers.data(name='X', shape=[1], dtype='float32')
   y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)


