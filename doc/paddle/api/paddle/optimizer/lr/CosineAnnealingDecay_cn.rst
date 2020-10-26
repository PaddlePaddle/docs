.. _cn_api_paddle_optimizer_lr_CosineAnnealingDecay:

CosineAnnealingDecay
-----------------------------------

.. py:class:: paddle.optimizer.lr.CosineAnnealingDecay(learning_rate, T_max, eta_min=0, last_epoch=-1, verbose=False) 

该接口使用 ``cosine annealing`` 的策略来动态调整学习率。

.. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}


:math:`\eta_{max}` 的初始值为 ``learning_rate``， :math:`T_{cur}` 是SGDR（重启训练SGD）训练过程中的当前训练轮数。SGDR的训练方法可以参考文档 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_.
这里只是实现了 ``cosine annealing`` 动态学习率，热启训练部分没有实现。 


参数：
    - **learning_rate** (float) - 初始学习率，也就是公式中的 :math:`\eta_{max}` ，数据类型为Python float。
    - **T_max** (float|int) - 训练的上限轮数，是余弦衰减周期的一半。
    - **eta_min** (float|int, 可选) - 学习率的最小值，即公式中的 :math:`\eta_{min}` 。默认值为0。 
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回：用于调整学习率的 ``CosineAnnealingDecay`` 实例对象。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dynamic graph mode
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.uniform([10, 10])
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
        scheduler.step()

    # train on static graph mode
    paddle.enable_static()
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[None, 4, 5])
        y = paddle.static.data(name='y', shape=[None, 4, 5])
        z = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(z)
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler)
        sgd.minimize(loss)

    exe = paddle.static.Executor()
    exe.run(start_prog)
    for epoch in range(20):
        for batch_id in range(2):
            out = exe.run(
                main_prog,
                feed={
                    'x': np.random.randn(3, 4, 5).astype('float32'),
                    'y': np.random.randn(3, 4, 5).astype('float32')
                },
                fetch_list=loss.name)
        scheduler.step()

.. py:method:: step(epoch=None)

step函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
  - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。

