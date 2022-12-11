.. _cn_api_fluid_dygraph_MultiStepDecay:

MultiStepDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.MultiStepDecay(learning_rate, milestones, decay_rate=0.1)



该接口提供 ``MultiStep`` 衰减学习率的功能。

算法可以描述为：

.. code-block:: text

    learning_rate = 0.5
    milestones = [30, 50]
    decay_rate = 0.1
    if epoch < 30:
        learning_rate = 0.5
    elif epoch < 50:
        learning_rate = 0.05
    else:
        learning_rate = 0.005

参数
::::::::::::

    - **learning_rate** (float|int) - 初始化的学习率。可以是Python的float或int。
    - **milestones** (tuple|list) - 列表或元组。必须是递增的。
    - **decay_rate** (float, optional) - 学习率的衰减率。``new_lr = origin_lr * decay_rate``。其值应该小于1.0。默认：0.1。

返回
::::::::::::
 无

代码示例
::::::::::::

    .. code-block:: python
        
        import paddle.fluid as fluid
        import numpy as np
        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)
            scheduler = fluid.dygraph.MultiStepDecay(0.5, milestones=[3, 5])
            adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())
            for epoch in range(6):
                for batch_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    adam.minimize(loss)
                scheduler.epoch()
                print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                # epoch:0, current lr is 0.5
                # epoch:1, current lr is 0.5
                # epoch:2, current lr is 0.5
                # epoch:3, current lr is 0.05
                # epoch:4, current lr is 0.05
                # epoch:5, current lr is 0.005

方法
::::::::::::
epoch(epoch=None)
'''''''''
通过当前的 epoch 调整学习率，调整后的学习率将会在下一次调用 ``optimizer.minimize`` 时生效。

**参数**

  - **epoch** (int|float，可选) - 类型：int或float。指定当前的epoch数。默认：无，此时将会自动累计epoch数。

**返回**

    无

**代码示例**

    参照上述示例代码。
