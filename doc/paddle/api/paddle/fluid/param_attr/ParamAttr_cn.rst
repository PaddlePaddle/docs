.. _cn_api_fluid_ParamAttr:


ParamAttr
-------------------------------


.. py:class:: paddle.fluid.ParamAttr(name=None, initializer=None, learning_rate=1.0, regularizer=None, trainable=True, do_model_average=False, need_clip=True)




.. note::
    该类中的 ``gradient_clip`` 属性在2.0版本会废弃，推荐使用``need_clip``来设置梯度裁剪范围，并在初始化 ``optimizer`` 时设置梯度裁剪。
    共有三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、:ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。

创建一个参数属性对象，用户可设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性。

参数:
    - **name** (str，可选) - 参数的名称。默认值为None，表示框架自动创建参数的名称。
    - **initializer** (Initializer，可选) - 参数的初始化方式。默认值为None，表示权重参数采用Xavier初始化方式，偏置参数采用全0初始化方式。
    - **learning_rate** (float) - 参数的学习率。实际参数的学习率等于全局学习率乘以参数的学习率，再乘以learning rate schedule的系数。
    - **regularizer** (WeightDecayRegularizer，可选) - 正则化方法。支持两种正则化策略: :ref:`cn_api_fluid_regularizer_L1Decay` 、 
      :ref:`cn_api_fluid_regularizer_L2Decay` ，如果在 ``optimizer`` (例如 :ref:`cn_api_fluid_optimizer_SGDOptimizer` ) 中也
      设置了正则化，``optimizer`` 中的正则化将被忽略。默认值为None，表示没有正则化。
    - **trainable** (bool) - 参数是否需要训练。默认值为True，表示需要训练。
    - **do_model_average** (bool) - 是否做模型平均。默认值为False，表示不做模型平均。
    - **need_clip** (bool) - 参数是否需要进行梯度裁剪。默认值为True，表示该参数的梯度会根据优化器中设置的裁剪规则进行裁剪。

返回: 表示参数属性的对象。

返回类型: ParamAttr

**代码示例**

.. code-block:: python

   import paddle
   
   weight_attr = paddle.ParamAttr(name="weight",
                                  learning_rate=0.5,
                                  regularizer=paddle.regularizer.L2Decay(1.0),
                                  trainable=True)
   print(weight_attr.name) # "weight"
   paddle.nn.Linear(3, 4, weight_attr=weight_attr)


