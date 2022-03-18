.. _cn_api_paddle_optimizer_lr_CyclicLR:

CyclicLR
-----------------------------------

.. py:class:: paddle.optimizer.lr.CyclicLR(base_learning_rate, max_learning_rate, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, last_epoch, verbose)

该接口提供一种学习率按固定频率在两个边界之间循环的策略。

内置了三种学习率缩放策略，分别如下：
    - **triangular**： 没有任何缩放的三角循环
    - **triangular2**：每个三角循环里将初始幅度缩放一半。
    - **exp_range**：每个循环中将初始幅度按照指数函数进行缩放，公式为 :math:`gamma^{cycle iterations}`。

初始幅度由`max_learning_rate - base_learning_rate`定义，:math:`gamma`为一个常量，:math:`cycle iterations`表示`cycle`数或'iterations'数。
cycle定义为:math:`cycle = floor(1 + epoch / (step_size_up + step_size_down)), 需要注意的是，CyclicLR应在每个批次的训练后调用step，因此这里的epoch表示当前实际迭代数，iterations则表示从训练开始时到当前时刻的迭代数量。

参数：
    - **base_learning_rate** (float) - 初始学习率，也是学习率变化的下边界。
    - **max_learning_rate** (float) - 最大学习率，需要注意的是，实际的学习率由``base_learning_rate``与初始幅度的缩放求和而来，因此实际学习率可能达不到``max_learning_rate``。
    - **step_size_up** (int) - 学习率从初始学习率增长到最大学习率所需步数。
    - **step_size_down** (int，可选) - 学习率从最大学习率下降到初始学习率所需步数。若未指定，则其值默认等于``step_size_up``。
    - **mode** (str) - 可以是'triangular'、'triangular2'或者'exp_range'，对应策略已在上文描述，当`scale_fn`被指定时时，该参数将被忽略。默认值：'triangular'。
    - **gamma** (float) - 'exp_range'缩放函数中的常量。
    - **sacle_fn** (function, 可选) - 一个有且仅有单个参数的函数，且对于任意的输入x，都必须满足0 <= scale_fn(x) <= 1；如果该参数被指定，则会忽略`mode`参数。
    - **scale_mode** (str) - 'cycle'或者'iterations'，表示缩放函数使用`cycle`数或`iterations`数作为输入。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回：用于调整学习率的``CyclicLR``实例对象。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dynamic graph mode
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr.CyclicLR(learning_rate=0.5, T_max=10, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(5):
            x = paddle.uniform([10, 10])
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
            scheduler.step()    # you should update learning rate each step
        

    # train on static graph mode
    paddle.enable_static()
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[None, 4, 5])
        y = paddle.static.data(name='y', shape=[None, 4, 5])
        z = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(z)
        scheduler = paddle.optimizer.lr.CyclicLR(learning_rate=0.5, T_max=10, verbose=True)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler)
        sgd.minimize(loss)

    exe = paddle.static.Executor()
    exe.run(start_prog)
    for epoch in range(20):
        for batch_id in range(5):
            out = exe.run(
                main_prog,
                feed={
                    'x': np.random.randn(3, 4, 5).astype('float32'),
                    'y': np.random.randn(3, 4, 5).astype('float32')
                },
                fetch_list=loss.name)
            scheduler.step()    # you should update learning rate each step

.. py:method:: step(epoch=None)

step函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
  - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。