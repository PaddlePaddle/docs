.. _cn_api_fluid_dygraph_LambdaDecay:

LambdaDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.LambdaDecay(learning_rate, lr_lambda)



该API提供 lambda函数 设置学习率的功能。``lr_lambda`` 为一个lambda函数，其通过 ``epoch`` 计算出一个因子，该因子会乘以初始学习率。

算法可以描述为：

.. code-block:: text

    learning_rate = 0.5        # init learning_rate
    lr_lambda = lambda epoch: 0.95 ** epoch
    
    learning_rate = 0.5        # epoch 0
    learning_rate = 0.475      # epoch 1
    learning_rate = 0.45125    # epoch 2

参数
::::::::::::

    - **learning_rate** (float|int) - 初始化的学习率。可以是Python的float或int。
    - **lr_lambda** (function) - ``lr_lambda`` 为一个lambda函数，其通过 ``epoch`` 计算出一个因子，该因子会乘以初始学习率。

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
            scheduler = fluid.dygraph.LambdaDecay(0.5, lr_lambda=lambda x: 0.95**x)
            adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())
            for epoch in range(6):
                for batch_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    adam.minimize(loss)
                scheduler.epoch()
                print("epoch:%d, current lr is %f" .format(epoch, adam.current_step_lr()))
                # epoch:0, current lr is 0.5
                # epoch:1, current lr is 0.475
                # epoch:2, current lr is 0.45125

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
