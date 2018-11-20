

.. _cn_api_fluid_clip_ErrorClipByValue:

ErrorClipByValue
>>>>>>>>>>>>

 .. py:class:: paddle.fluid.clip.ErrorClipByValue(max, min=None)

将张量值的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 `max` 之间

  - 任何小于最小值的值都被设置为最小值

  - 任何大于max的值都被设置为max

参数:

  - **max** (foat) - 要修剪的最大值。

  - **min** (float) - 要修剪的最小值。如果用户没有设置，将被设置为框架-max。
  
**代码示例**
 
.. code-block:: python
        
     var = fluid.framework.Variable(..., error_clip=ErrorClipByValue(max=5.0), ...)


.. _cn_api_fluid_clip_GradientClipByValue:

GradientClipByValue
>>>>>>>>>>>>

 .. py:class:: paddle.fluid.clip.GradientClipByValue(max, min=None)

将张量值的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 `max` 之间

  - 任何小于最小值的值都被设置为最小值

  - 任何大于max的值都被设置为max

参数:

  - **max** (foat) - 要修剪的最大值。

  - **min** (float，optional) - 要修剪的最小值。如果用户没有设置，将被设置为框架-max。
  
**代码示例**
 
.. code-block:: python
        
     w_param_attrs = ParamAttr(name=None,
     initializer=UniformInitializer(low=-1.0, high=1.0, seed=0),
     learning_rate=1.0,
     regularizer=L1Decay(1.0),
     trainable=True,
     clip=GradientClipByValue(-1.0, 1.0))
     y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)
     
.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
>>>>>>>>>>>>

 .. py:class:: paddle.fluid.clip.GradientClipByNorm(clip_norm)

将张量转换为L2范数不超过 ``clip_norm`` 的张量

该operator 限制了 输入张量 ``X``的L2范数不会超过 max_norm 。如果 ``X`` 的 ``L2`` 范数小于或等于 ``max_norm`` ,输出和 ``X`` 一样，如果 ``X`` 的L2范数大于 ``max_norm`` , ``X`` 将被线性缩放到L2范数等于 ``max_norm`` ,如以下公式所示:
.. math::
            \\Out = \frac{max\_norm∗X}{norm(X)}\\

其中 ``norm（X）`` 代表 ``X`` 的 L2 范数


参数:

  - **clip_norm** (float) - 二范数最大值

  
**代码示例**
 
.. code-block:: python
        
    w_param_attrs = ParamAttr(name=None,
    initializer=UniformInitializer(low=-1.0, high=1.0, seed=0),
    learning_rate=1.0,
    regularizer=L1Decay(1.0),
    trainable=True,
    clip=GradientClipByNorm(clip_norm=2.0))
    y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)


