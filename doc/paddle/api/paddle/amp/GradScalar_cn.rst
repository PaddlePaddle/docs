.. _cn_api_amp_GradScalar:

GradScaler
-------------------------------

.. py:class:: paddle.amp.GradScaler(enable=True, init_loss_scaling=2.**15, incr_ratio=2.0, decr_ratio=0.5, incr_every_n_steps=1000, decr_every_n_nan_or_inf=1, use_dynamic_loss_scaling=True)



GradScaler用于动态图模式下的"自动混合精度"的训练或推断。它控制loss的缩放比例，有助于避免浮点数溢出的问题。这个类具有 ``scale()`` 和 ``minimize()`` 两个方法。

``scale()`` 用于让loss乘上一个缩放的比例。
``minimize()`` 与 ``Optimizer.minimize()`` 类似，执行参数的更新。

通常，GradScaler和 ``paddle.amp.auto_cast`` 一起使用，来实现动态图模式下的"自动混合精度"。


参数：
    - **enable** (bool, 可选) - 是否使用loss scaling。默认值为True。
    - **init_loss_scaling** (float, 可选) - 初始loss scaling因子。默认值为2**15。
    - **incr_ratio** (float, 可选) - 增大loss scaling时使用的乘数。默认值为2.0。
    - **decr_ratio** (float, 可选) - 减小loss scaling时使用的小于1的乘数。默认值为0.5。
    - **incr_every_n_steps** (int, 可选) - 连续n个steps的梯度都是有限值时，增加loss scaling。默认值为1000。
    - **decr_every_n_nan_or_inf** (int, 可选) - 累计出现n个steps的梯度为nan或者inf时，减小loss scaling。默认值为2。
    - **use_dynamic_loss_scaling** (bool, 可选) - 是否使用动态的loss scaling。如果不使用，则使用固定的loss scaling；如果使用，则会动态更新loss scaling。默认值为True。

返回：
    一个AmpScaler对象。


**代码示例**：

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2d(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
        scaled = scaler.scale(loss)  # scale the loss 
        scaled.backward()            # do backward
        scaler.minimize(optimizer, scaled)  # update parameters


.. py:function:: scale(var)

将Tensor乘上缩放因子，返回缩放后的输出。
如果这个 :class:`GradScaler` 的实例不使用loss scaling，则返回的输出将保持不变。

参数：
    - **var** (Tensor) - 缩放的Tensor。

返回：缩放后的Tensor或者原Tensor。

代码示例：

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2d(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
        scaled = scaler.scale(loss)  # scale the loss 
        scaled.backward()            # do backward
        scaler.minimize(optimizer, scaled)  # update parameters

.. py:function:: minimize(optimizer, *args, **kwargs)

这个函数与 ``Optimizer.minimize()`` 类似，用于执行参数更新。
如果参数缩放后的梯度包含NAN或者INF，则跳过参数更新。否则，首先让缩放过梯度的参数取消缩放，然后更新参数。
最终，更新loss scaling的比例。

参数：
    - **optimizer** (Optimizer) - 用于更新参数的优化器。
    - **args** - 参数，将会被传递给 ``optimizer.minimize()`` 。
    - **kwargs** - 关键词参数，将会被传递给 ``Optimizer.minimize()`` 。

代码示例：

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2d(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
        scaled = scaler.scale(loss)  # scale the loss 
        scaled.backward()            # do backward
        scaler.minimize(optimizer, scaled)  # update parameters






