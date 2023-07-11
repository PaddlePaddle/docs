.. _cn_api_paddle_optimizer_RMSProp:

RMSProp
-------------------------------

.. py:class:: paddle.optimizer.RMSProp(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, parameters=None, weight_decay=None, grad_clip=None, name=None)

均方根传播（RMSProp）法，是一种未发表的，自适应学习率的方法。原演示幻灯片中的第 29 张提出了 `RMSProp <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ 。等式如下所示：

.. math::
    r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
    w & = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)

第一个等式计算每个权重平方梯度的移动平均值，然后将梯度除以 :math:`sqrtv（w，t）` 。

.. math::
   r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
   v(w, t) & = \beta v(w, t-1) +\frac{\eta} {\sqrt{r(w,t) +\epsilon}} \nabla Q_{i}(w)\\
         w & = w - v(w, t)

如果居中为真：

.. math::
      r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
      g(w, t) & = \rho g(w, t-1) + (1 -\rho)\nabla Q_{i}(w)\\
      v(w, t) & = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +\epsilon}} \nabla Q_{i}(w)\\
            w & = w - v(w, t)

其中，:math:`ρ` 是超参数，典型值为 0.9,0.95 等。:math:`beta` 是动量术语。:math:`epsilon` 是一个平滑项，用于避免除零，通常设置在 1e-4 到 1e-8 的范围内。

参数
::::::::::::

    - **learning_rate** (float) - 全局学习率。
    - **rho** (float，可选) - rho 是等式中的 :math:`rho`，默认值 0.95。
    - **epsilon** (float，可选) - 等式中的 epsilon 是平滑项，避免被零除，默认值 1e-6。
    - **momentum** (float，可选) - 方程中的β是动量项，默认值 0.0。
    - **centered** (bool，可选) - 如果为 True，则通过梯度的估计方差，对梯度进行归一化；如果 False，则由未 centered 的第二个 moment 归一化。将此设置为 True 有助于模型训练，但会消耗额外计算和内存资源。默认为 False。
    - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **weight_decay** (float|WeightDecayRegularizer，可选) - 正则化方法。可以是 float 类型的 L2 正则化系数或者正则化策略：:ref:`cn_api_fluid_regularizer_L1Decay` 、
      :ref:`cn_api_fluid_regularizer_L2Decay`。如果一个参数已经在 :ref:`cn_api_fluid_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
      如果没有在 :ref:`cn_api_fluid_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为 None，表示没有正则化。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_fluid_clip_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_fluid_clip_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_fluid_clip_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


代码示例
::::::::::::

.. code-block:: python

    import paddle

    inp = paddle.rand([10,10], dtype="float32")
    linear = paddle.nn.Linear(10, 10)
    out = linear(inp)
    loss = paddle.mean(out)

    rmsprop = paddle.optimizer.RMSProp(learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01)
    out.backward()
    rmsprop.step()
    rmsprop.clear_grad()

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
    a = paddle.rand([2,13], dtype="float32")
    linear = paddle.nn.Linear(13, 5)
    rmsprop = paddle.optimizer.RMSProp(learning_rate = 0.01,
                                parameters = linear.parameters())
    out = linear(a)
    out.backward()
    rmsprop.step()
    rmsprop.clear_grad()

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

    inp = paddle.rand([10,10], dtype="float32")
    linear = paddle.nn.Linear(10, 10)
    out = linear(inp)
    loss = paddle.mean(out)

    rmsprop = paddle.optimizer.RMSProp(learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01)
    out.backward()
    rmsprop.step()
    rmsprop.clear_grad()

clear_gradients()
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

.. code-block:: python

    import paddle

    a = paddle.rand([2,13], dtype="float32")
    linear = paddle.nn.Linear(13, 5)
    rmsprop = paddle.optimizer.RMSProp(learning_rate=0.02,
                                     parameters=linear.parameters())
    out = linear(a)
    out.backward()
    rmsprop.step()
    rmsprop.clear_gradients()

set_lr(value)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

手动设置当前 ``optimizer`` 的学习率。当使用_LRScheduler 时，无法使用该 API 手动设置学习率，因为这将导致冲突。

**参数**

    value (float) - 需要设置的学习率的值。

**返回**

无。

**代码示例**

.. code-block:: python


    import paddle

    linear = paddle.nn.Linear(10, 10)
    rmsprop = paddle.optimizer.RMSProp(0.1, parameters=linear.parameters())

    # set learning rate manually by python float value
    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(5):
        rmsprop.set_lr(lr_list[i])
        lr = rmsprop.get_lr()
        print("current lr is {}".format(lr))
    # Print:
    #    current lr is 0.2
    #    current lr is 0.3
    #    current lr is 0.4
    #    current lr is 0.5
    #    current lr is 0.6

set_lr_scheduler(scheduler)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

手动设置当前 ``optimizer`` 的学习率为 LRScheduler 类。

**参数**

    scheduler (LRScheduler) - 需要设置的学习率的 LRScheduler 类。

**返回**

无。

**代码示例**

.. code-block:: python
    import paddle
    linear = paddle.nn.Linear(10, 10)
    rmsprop = paddle.optimizer.RMSProp(0.1, parameters=linear.parameters())
    # set learning rate manually by class LRScheduler
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2,4,6], gamma=0.8)
    rmsprop.set_lr_scheduler(scheduler)
    lr = rmsprop.get_lr()
    print("current lr is {}".format(lr))
    #    current lr is 0.5
    # set learning rate manually by another LRScheduler
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=5, gamma=0.6)
    rmsprop.set_lr_scheduler(scheduler)
    lr = rmsprop.get_lr()
    print("current lr is {}".format(lr))
    #    current lr is 0.1

get_lr()
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    # example1: _LRScheduler is not used, return value is all the same
    emb = paddle.nn.Embedding(10, 10, sparse=False)
    rmsprop = paddle.optimizer.RMSProp(0.001, parameters = emb.parameters())
    lr = rmsprop.get_lr()
    print(lr) # 0.001

    # example2: StepDecay is used, return the step learning rate
    linear = paddle.nn.Linear(10, 10)
    inp = paddle.rand([10,10], dtype="float32")
    out = linear(inp)
    loss = paddle.mean(out)

    bd = [2, 4, 6, 8]
    value = [0.2, 0.4, 0.6, 0.8, 1.0]
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
    rmsprop = paddle.optimizer.RMSProp(scheduler,
                           parameters=linear.parameters())

    # first step: learning rate is 0.2
    np.allclose(rmsprop.get_lr(), 0.2, rtol=1e-06, atol=0.0) # True

    # learning rate for different steps
    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
    for i in range(12):
        rmsprop.step()
        lr = rmsprop.get_lr()
        scheduler.step()
        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True
