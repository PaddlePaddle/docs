.. _cn_api_paddle_optimizer_Momentum:

Momentum
-------------------------------

.. py:class:: paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=None, use_nesterov=False, weight_decay=None, grad_clip=None, name=None)


该接口实现含有速度状态的Simple Momentum 优化器

该优化器含有牛顿动量标志，公式更新如下：

更新公式如下：


.. math::
    & velocity = mu * velocity + gradient\\
    & if (use\_nesterov):\\
    &\quad   param = param - (gradient + mu * velocity) * learning\_rate\\
    & else:\\&\quad   param = param - learning\_rate * velocity


参数：
    - **learning_rate** (float|_LRScheduler) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler类，默认值为0.001
    - **momentum** (float) - 动量因子。
    - **parameters** (list, 可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为None，这时所有的参数都将被优化。
    - **use_nesterov** (bool, 可选) - 赋能牛顿动量，默认值False。
    - **weight_decay** (float|Tensor, 可选) - 权重衰减系数，是一个float类型或者shape为[1] ，数据类型为float32的Tensor类型。默认值为0.01
    - **grad_clip** (GradientClipBase, 可选) – 梯度裁剪的策略，支持三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
      默认值为None，此时将不进行梯度裁剪。
    - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None



**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)
    momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out.backward()
    momentum.step()
    momentum.clear_grad()


.. py:method:: step()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

执行一次优化器并进行参数更新。

返回：None。



**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=0.0003, parameters = linear.parameters())
    out = linear(a)
    out.backward()
    momentum.step()
    momentum.clear_grad()

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

    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)

    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out.backward()
    momentum.minimize(loss)
    momentum.clear_grad()

.. py:method:: clear_grad()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)
    linear = paddle.nn.Linear(13, 5)
    optimizer = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
    out = linear(a)
    out.backward()
    optimizer.step()
    optimizer.clear_grad()

.. py:method:: set_lr(value)

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**  
 

