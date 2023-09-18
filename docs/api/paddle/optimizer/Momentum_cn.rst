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

    - **learning_rate** (float|_LRScheduler，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler 类的 tensor ，默认值为 0.001。
    - **momentum** (float，可选) - 动量因子。默认值为0.001。
    - **parameters** (list|tuple，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数。您可以为不同的参数组指定不同的选项，如学习率，权重衰减等，参数是dict的列表。注意，参数组中的learning_rate表示基础值的learning_rate。在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **use_nesterov** (bool，可选) - 赋能牛顿动量，默认值 False。
    - **weight_decay** (float|Tensor，可选) - 权重衰减系数，是一个 float 类型或者 shape 为[1]，数据类型为 float32 的 Tensor 类型。默认值为 0.01。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **multi_precision** (bool，可选) - 在权重更新时是否使用多精度。默认值 False。
    - **rescale_grad**(float,可选) - 用梯度 rescale_grad 之前更新。通常选择为1.0/batch_size。
    - **use_multi_tensor** (bool，可选) - 是否使用 multi-tensor 策略一次性更新所有参数。默认值 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



代码示例
::::::::::::

COPY-FROM: paddle.optimizer.Momentum


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

COPY-FROM: paddle.optimizer.Momentum.step

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

COPY-FROM: paddle.optimizer.Momentum.minimize

clear_grad()
'''''''''

.. note::

 该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

COPY-FROM: paddle.optimizer.Momentum.clear_grad

set_lr(value)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。在优化器中手动设置学习率的值。如果优化器使用LRScheduler,不能调用此API,因为它会导致冲突。

set_lr_scheduler(scheduler)
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。
