.. _cn_api_fluid_layers_cosine_decay:

cosine_decay
-------------------------------

.. py:function:: paddle.fluid.layers.cosine_decay(learning_rate, step_each_epoch, epochs)

:alias_main: paddle.nn.functional.cosine_decay
:alias: paddle.nn.functional.cosine_decay,paddle.nn.functional.learning_rate.cosine_decay
:old_api: paddle.fluid.layers.cosine_decay



使用 cosine decay 的衰减方式进行学习率调整。

在训练模型时，建议一边进行训练一边降低学习率。 通过使用此方法，学习速率将通过如下cosine衰减策略进行衰减：

.. math::
    decayed\_lr = learning\_rate * 0.5 * (cos(epoch * math.pi / epochs) + 1)


参数：
    - **learning_rate** （Variable | float） - 初始学习率。
    - **step_each_epoch** （int） - 一次迭代中的步数。
    - **epochs**  - 总迭代次数。




**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    lr = fluid.layers.cosine_decay( learning_rate = base_lr, step_each_epoch=10000, epochs=120)



