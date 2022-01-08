.. _cn_api_paddle_optimizer_lr_MultiplicativeDecay:

MultiplicativeDecay
-----------------------------------

.. py:class:: paddle.optimizer.lr.MultiplicativeDecay(learning_rate, lr_lambda, last_epoch=-1, verbose=False)

该接口提供 ``lambda`` 函数设置学习率的策略。 ``lr_lambda`` 为一个 ``lambda`` 函数，其通过 ``epoch`` 计算出一个因子，该因子会乘以当前学习率。

衰减过程可以参考以下代码：

.. code-block:: text

    learning_rate = 0.5        # init learning_rate
    lr_lambda = lambda epoch: 0.95

    learning_rate = 0.5        # epoch 0,
    learning_rate = 0.475      # epoch 1, 0.5*0.95
    learning_rate = 0.45125    # epoch 2, 0.475*0.95
    ...


参数：
    - **learning_rate** （float） - 初始学习率，数据类型为Python float。
    - **lr_lambda** （function）- lr_lambda 为一个lambda函数，其通过 epoch 计算出一个因子，该因子会乘以当前学习率。
    - **last_epoch** （int，可选）- 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** （bool，可选）- 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回：用于调整学习率的 ``MultiplicativeDecay`` 实例对象。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dynamic graph mode
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=0.5, lr_lambda=lambda x:0.95, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.uniform([10, 10])
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.minimize(loss)
            linear.clear_gradients()
            scheduler.step()    # If you update learning rate each step
      # scheduler.step()        # If you update learning rate each epoch

.. py:method:: step(epoch=None)

step函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
  - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。

