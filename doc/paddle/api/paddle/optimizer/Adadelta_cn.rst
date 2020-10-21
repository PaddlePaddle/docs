.. _cn_api_paddle_optimizer_Adadelta:

AdadeltaOptimizer
-------------------------------

.. py:class:: paddle.optimizer.Adadelta(learning_rate, epsilon=1.0e-6, rho=0.95, parameter_list=None, regularization=None, grad_clip=None, name=None)


**注意：此接口不支持稀疏参数更新。**

Adadelta优化器，具体细节可参考论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

更新公式如下：

.. math::

    E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2\\
    learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
    E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2


参数：
    - **learning_rate** (float|_LRScheduler) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler类，默认值为0.001
    - **epsilon** (float, 可选) - 保持数值稳定性的短浮点类型值，默认值为1e-06
    - **rho** (float) - 算法中的衰减率，默认值为0.95。
    - **parameters** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **weight_decay** (float|Tensor, 可选) - 权重衰减系数，是一个float类型或者shape为[1] ，数据类型为float32的Tensor类型。默认值为0.01
    - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
      默认值为None，此时将不进行梯度裁剪。
    - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

Adadelta优化器出自 `DECOUPLED WEIGHT DECAY REGULARIZATION 论文 <https://arxiv.org/pdf/1711.05101.pdf>`,用来解决Adam优化器中L2正则化失效的问题。



**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)
    adadelta = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95,
            parameters=linear.parameters())
    out.backward()
    adadelta.step()
    adadelta.clear_grad()


.. py:method:: step()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

执行一次优化器并进行参数更新。

返回：None。



**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    paddle.disable_static()
    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)
    linear = paddle.nn.Linear(13, 5)
    adadelta = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95,
                                parameters = linear.parameters())
    out = linear(a)
    out.backward()
    adadelta.step()
    adadelta.clear_grad()

.. py:method:: minimize(loss, startup_program=None, parameters=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameters中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Tensor) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameters中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameters** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的集合，默认值为None
         
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)

    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    adadelta = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95,
            parameters=linear.parameters())
    out.backward()
    adadelta.minimize(loss)
    adadelta.clear_grad()

.. py:method:: clear_grad()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)
    linear = paddle.nn.Linear(13, 5)
    optimizer = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95,
                                     parameters=linear.parameters())
    out = linear(a)
    out.backward()
    optimizer.step()
    optimizer.clear_grad()

.. py:method:: set_lr(value)

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**  

手动设置当前 ``optimizer`` 的学习率。当使用_LRScheduler时，无法使用该API手动设置学习率，因为这将导致冲突。

参数：
    value (float) - 需要设置的学习率的值。

返回：None

**代码示例**

.. code-block:: python

    import paddle
    paddle.disable_static()
    linear = paddle.nn.Linear(10, 10)

    adadelta = paddle.optimizer.AdamW(weight_decay=0.01,
                                 learning_rate=0.1, parameters=linear.parameters())

    # set learning rate manually by python float value
    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(5):
        adadelta.set_lr(lr_list[i])
        lr = adadelta.get_lr()
        print("current lr is {}".format(lr))
    # Print:
    #    current lr is 0.2
    #    current lr is 0.3
    #    current lr is 0.4
    #    current lr is 0.5
    #    current lr is 0.6

.. py:method:: get_lr()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

获取当前步骤的学习率。当不使用_LRScheduler时，每次调用的返回值都相同，否则返回当前步骤的学习率。

返回：float，当前步骤的学习率。


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    # example1: _LRScheduler is not used, return value is all the same
    paddle.disable_static()
    emb = paddle.nn.Embedding(10, 10, sparse=False)
    adadelta = paddle.optimizer.Adadelta(learning_rate=0.001, parameters = emb.parameters(),weight_decay=0.01)
    lr = adadelta.get_lr()
    print(lr) # 0.001

    # example2: PiecewiseDecay is used, return the step learning rate
    paddle.disable_static()
    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)

    bd = [2, 4, 6, 8]
    value = [0.2, 0.4, 0.6, 0.8, 1.0]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(bd, value, 0)
    adadelta = paddle.optimizer.Adadelta(scheduler,
                           parameters=linear.parameters(),
                           weight_decay=0.01)

    # first step: learning rate is 0.2
    np.allclose(adadelta.get_lr(), 0.2, rtol=1e-06, atol=0.0) # True

    # learning rate for different steps
    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
    for i in range(12):
        adadelta.step()
        lr = adadelta.get_lr()
        scheduler.step()
        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

