.. _cn_api_paddle_optimizer_LambdaLR:

LambdaLR
-----------------------------------

.. py:class:: paddle.optimizer.lr_scheduler.LambdaLR(learning_rate, lr_lambda, last_epoch=-1, verbose=False)

该接口提供 ``lambda`` 函数设置学习率的功能。 ``lr_lambda`` 为一个 ``lambda`` 函数，其通过 ``epoch`` 计算出一个因子，该因子会乘以初始学习率。。

衰减过程可以参考以下代码：

.. code-block:: python

    learning_rate = 0.5        # init learning_rate
    lr_lambda = lambda epoch: 0.95 ** epoch
    learning_rate = 0.5        # epoch 0
    learning_rate = 0.475      # epoch 1
    learning_rate = 0.45125    # epoch 2


参数
:::::::::
    - **learning_rate** （float） - 初始学习率，数据类型为Python float。
    - **lr_lambda** （function）：lr_lambda 为一个lambda函数，其通过 epoch 计算出一个因子，该因子会乘以初始学习率。
    - **last_epoch** （int，可选）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率 。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
:::::::::
    返回计算LambdaLR的可调用对象。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dygraph mode
    paddle.disable_static()
    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr_scheduler.LambdaLR(learning_rate=0.5, lr_lambda=lambda x:0.95**x, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameter_list=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.to_tensor(x)
            out = linear(x)
            loss = paddle.reduce_mean(out)
            loss.backward()
            sgd.minimize(loss)
            linear.clear_gradients()
        scheduler.step()

    # train on static mode
    paddle.enable_static()
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[None, 4, 5])
        y = paddle.static.data(name='y', shape=[None, 4, 5])
        z = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(z)
        scheduler = paddle.optimizer.lr_scheduler.LambdaLR(learning_rate=0.5, lr_lambda=lambda x:0.95**x, verbose=True)
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

step函数需要在优化器的 `step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
  - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。

