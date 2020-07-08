.. _cn_api_fluid_dygraph_CosineAnnealingDecay:

CosineAnnealingDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.CosineAnnealingDecay(learning_rate, T_max, eta_min)

:api_attr: 命令式编程模式（动态图)

该API提供 CosineAnnealing 设置学习率的功能。 其中 :math:`\eta_{max}` 为初始学习率。 :math:`T_{cur}` 为SGDR自上次重启动后经过的 ``epoch`` 数。

算法可以描述为：

.. math::
    \begin{aligned}
        \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
        + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
        & T_{cur} \neq (2k+1)T_{max}; \\
        \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
        \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
        & T_{cur} = (2k+1)T_{max}.
    \end{aligned}

该策略是在论文 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_ 中提出。
注意，这里只实现了SGDR的 CosineAnnealing 部分，没有实现重启。

参数：
    - **learning_rate** (float|int) - 初始化的学习率，也就是 :math:`\eta_{max}` 。可以是Python的float或int。
    - **T_max** (int) - 最大迭代次数。它是学习率变化周期的1/2。
    - **eta_min** (float|int, optional) - 最小学习率，也就是 :math:`\eta_{min}` 。默认：0.

返回： 无

**代码示例**：

    .. code-block:: python
               
        import paddle.fluid as fluid
        import numpy as np
        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)
            scheduler = fluid.dygraph.CosineAnnealingDecay(0.5, 5)
            adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())
            for epoch in range(10):
                for batch_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    adam.minimize(loss)
                scheduler.epoch()
                print("epoch:%d, current lr is %f" % (epoch, adam.current_step_lr()))
                # epoch:0, current lr is 0.5        (eta_max) The beginning of 1th cycle
                # epoch:1, current lr is 0.452254
                # epoch:2, current lr is 0.327254
                # ...
                # epoch:5, current lr is 0.095492   half of 1th cycle
                # ...
                # epoch:10, current lr is 0.5       (eta_max) The end of 1th cycle
                # ...

.. py:method:: epoch(epoch=None)
通过当前的 epoch 调整学习率，调整后的学习率将会在下一次调用 ``optimizer.minimize`` 时生效。

参数：
  - **epoch** (int|float，可选) - 类型：int或float。指定当前的epoch数。默认：无，此时将会自动累计epoch数。

返回：
    无

**代码示例**:

    参照上述示例代码。
