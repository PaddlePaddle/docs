.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByNorm(clip_norm)

将张量转换为L2范数不超过 ``clip_norm`` 的张量

该operator 限制了 输入张量 :math:`X` 的L2范数不会超过 :math:`max\_norm` 。如果 :math:`X` 的 ``L2`` 范数小于或等于 :math:`max\_norm` ,输出和 :math:`X` 一样，如果 :math:`X` 的L2范数大于 :math:`max\_norm` , :math:`X` 将被线性缩放到L2范数等于 :math:`max\_norm` ,如以下公式所示:

.. math::
            \\Out = \frac{max\_norm∗X}{norm(X)}\\

其中 :math:`norm（X）` 代表 :math:`X` 的 L2 范数


参数:
 - **clip_norm** (float) - 二范数最大值

  
**代码示例**
 
.. code-block:: python
        
    import paddle.fluid as fluid
    w_param_attrs = fluid.ParamAttr(name=None,
                                    initializer=fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0),
                                    learning_rate=1.0,
                                    regularizer=fluid.regularizer.L1Decay(1.0),
                                    trainable=True,
                                    gradient_clip=fluid.clip.GradientClipByNorm(clip_norm=2.0))
    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)








