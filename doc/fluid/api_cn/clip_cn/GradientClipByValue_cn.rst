.. _cn_api_fluid_clip_GradientClipByValue:

GradientClipByValue
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByValue(max, min=None)

将梯度值(gradient values)的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 ``max`` 之间

- 任何小于最小值的值都被设置为最小值

- 任何大于max的值都被设置为max

参数:
 - **max** (foat) - 要修剪的最大值。
 - **min** (float，optional) - 要修剪的最小值。如果用户没有设置，将被 ``framework`` 设置为 ``-max`` 。
  
**代码示例**
 
.. code-block:: python
        
     import paddle.fluid as fluid
     w_param_attrs = fluid.ParamAttr(name=None,
                                     initializer=fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0),
                                     learning_rate=1.0,
                                     regularizer=fluid.regualrizer.L1Decay(1.0),
                                     trainable=True,
                                     gradient_clip=fluid.clip.GradientClipByValue(-1.0, 1.0))
     x = fluid.layers.data(name='x', shape=[10], dtype='float32')
     y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)
     






