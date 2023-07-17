.. _cn_api_paddle_optimizer_Momentum:

Momentum
-------------------------------

.. py:class:: paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=None, use_nesterov=False, weight_decay=None, grad_clip=None, name=None)


含有速度状态的 Simple Momentum 优化器。

该优化器含有牛顿动量标志，公式更新如下：

更新公式如下：


.. math::
    & velocity = mu * velocity + gradient\\
    & if (use\_nesterov):\\
    &\quad   param = param - (gradient + mu * velocity) * learning\_rate\\
    & else:\\&\quad   param = param - learning\_rate * velocity


参数
::::::::::::

    - **learning_rate** (float|_LRScheduler，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler 类，默认值为 0.001。
    - **momentum** (float，可选) - 动量因子。
    - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **use_nesterov** (bool，可选) - 赋能牛顿动量，默认值 False。
    - **weight_decay** (float|Tensor，可选) - 权重衰减系数，是一个 float 类型或者 shape 为[1]，数据类型为 float32 的 Tensor 类型。默认值为 0.01。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_fluid_clip_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_fluid_clip_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_fluid_clip_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



代码示例
::::::::::::

.. code-block:: python

    import paddle

    inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 10)
    out = linear(inp)
    loss = paddle.mean(out)
    momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out.backward()
    momentum.step()
    momentum.clear_grad()


方法
::::::::::::
step()
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

执行一次优化器并进行参数更新。

**返回**

无。


**代码示例**

.. code-block:: python

    import paddle
    value = paddle.arange(26, dtype='float32')
    a = paddle.reshape(value, [2, 13])
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    out = linear(a)
    out.backward()
    momentum.step()
    momentum.clear_grad()

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters 中的 Parameters，最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量。
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_fluid_Program`，默认值为 None，此时将使用 :ref:`cn_api_fluid_default_startup_program`。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成的集合，默认值为 None。

**返回**

 tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

.. code-block:: python

    import paddle

    inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 10)
    out = linear(inp)
    loss = paddle.mean(out)

    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out.backward()
    momentum.minimize(loss)
    momentum.clear_grad()

clear_grad()
'''''''''

.. note::

 该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle

    value = paddle.arange(26, dtype='float32')
    a = paddle.reshape(value, [2, 13])
    linear = paddle.nn.Linear(13, 5)
    optimizer = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out = linear(a)
    out.backward()
    optimizer.step()
    optimizer.clear_grad()

set_lr(value)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

set_lr_scheduler(scheduler)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。
