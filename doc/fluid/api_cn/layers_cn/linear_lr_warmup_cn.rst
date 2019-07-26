.. _cn_api_fluid_layers_linear_lr_warmup:

linear_lr_warmup
-------------------------------

.. py:function:: paddle.fluid.layers.linear_lr_warmup(learning_rate, warmup_steps, start_lr, end_lr)

在正常学习率调整之前先应用线性学习率热身(warm up)进行初步调整。

.. code-block:: text

    if global_step < warmup_steps:
        linear_step = end_lr - start_lr
        lr = start_lr + linear_step * (global_step / warmup_steps)

参数：
    - **learning_rate** （float | Variable） - 学习率，类型为float值或变量。
    - **warmup_steps** （int） - 进行warm up过程的步数。
    - **start_lr** （float） - warm up的起始学习率
    - **end_lr** （float） - warm up的最终学习率。

返回：进行热身衰减后的学习率。


**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        boundaries = [100, 200]
        lr_steps = [0.1, 0.01, 0.001]
        warmup_steps = 50
        start_lr = 1. / 3.
        end_lr = 0.1
        decayed_lr = fluid.layers.linear_lr_warmup(
            fluid.layers.piecewise_decay(boundaries, lr_steps),
            warmup_steps, start_lr, end_lr)








