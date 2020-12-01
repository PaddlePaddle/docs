.. _cn_api_paddle_optimizer_lr_PiecewiseDecay:

PiecewiseDecay
-------------------------------

.. py:class:: paddle.optimizer.lr.PiecewiseDecay(boundaries, values, last_epoch=-1, verbose=False)


该接口提供分段设置学习率的策略。

过程可以描述如下：

.. code-block:: text

    boundaries = [100, 200]
    values = [1.0, 0.5, 0.1]

    learning_rate = 1.0     if epoch < 100
    learning_rate = 0.5     if 100 <= epoch < 200
    learning_rate = 0.1     if 200 <= epoch
    ...


参数：
    - **boundaries** (list) - 指定学习率的边界值列表。列表的数据元素为Python int类型。
    - **values** (list) - 学习率列表。数据元素类型为Python float的列表。与边界值列表有对应的关系。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率 。
    - **verbose** (bool，可选) - 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回：用于调整学习率的 ``PiecewiseDecay`` 实例对象。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dynamic graph mode
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[3, 6, 9], values=[0.1, 0.2, 0.3, 0.4], verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.uniform([10, 10])
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
            scheduler.step()    # If you update learning rate each step
      # scheduler.step()        # If you update learning rate each epoch

    # train on static graph mode
    paddle.enable_static()
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[None, 4, 5])
        y = paddle.static.data(name='y', shape=[None, 4, 5])
        z = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(z)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[3, 6, 9], values=[0.1, 0.2, 0.3, 0.4], verbose=True)
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
            scheduler.step()    # If you update learning rate each step
      # scheduler.step()        # If you update learning rate each epoch

.. py:method:: step(epoch=None)

step函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
  - **epoch** (int，可选) - 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。
