.. _cn_api_paddle_optimizer_Adadelta:

Adadelta
-------------------------------

.. py:class:: paddle.optimizer.Adadelta(learning_rate=0.001, epsilon=1e-06, rho=0.95, parameters=None, weight_decay=None, grad_clip=None, name=None)


.. note::
    此接口不支持稀疏参数更新。

Adadelta 优化器，是对 :ref:`Adagrad <cn_api_paddle_optimizer_Adagrad>` 的改进。

相关论文：`ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

更新公式如下：

.. math::

    E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2\\
    learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
    E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2


参数
::::::::::::

    - **learning_rate** (float|Tensor|LRScheduler，可选) - 学习率，用于参数更新的计算。可以是浮点值、浮点类型的 Tensor 或 LRScheduler。默认值为 0.001。
    - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为 1e-06。
    - **rho** (float，可选) - 算法中的衰减率，默认值为 0.95。
    - **parameters** (list|tuple|None，可选) - 指定优化器需要优化的参数，可以是待更新 Tensor 的列表或元组；也可以是参数组字典的列表，以为不同参数组指定学习率、权重衰减等选项。参数组中的 ``learning_rate`` 表示基础学习率的缩放比例。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，此时所有参数都将被优化。
    - **weight_decay** (int|float|WeightDecayRegularizer|None，可选) - 正则化方法。可以是 int 或 float 类型的 L2 正则化系数或者正则化策略：:ref:`cn_api_paddle_regularizer_L1Decay` 、:ref:`cn_api_paddle_regularizer_L2Decay`。如果参数已经在 :ref:`cn_api_paddle_ParamAttr` 中设置正则化，这里的设置将被忽略；否则该设置生效。默认值为 None，表示没有正则化。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.Adadelta


方法
::::::::::::
step(closure=None)
''''''''''''''''''''''''''''''''''''''''

.. note::
    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

执行一次优化器并进行参数更新。

**参数**

    - **closure** (Callable[[], Tensor], 可选) - 用于评估模型并返回损失的闭包函数。闭包函数应接受 0 个参数并返回 Tensor。适用于需要多次评估损失的优化过程。默认值为 None。

**返回**

Tensor 或 None。若传入 closure 参数则返回其输出的损失，否则返回 None。



COPY-FROM: paddle.optimizer.Adadelta.step

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters 中的 Parameters，最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_paddle_static_Program`，默认值为 None，此时将使用 :ref:`cn_api_paddle_static_default_startup_program` 。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成的集合，默认值为 None。

**返回**

 tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

COPY-FROM: paddle.optimizer.Adadelta.minimize

clear_grad(set_to_zero=True)
''''''''''''''''''''''''''''''''''''''''

.. note::
    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**参数**

    - **set_to_zero** (bool，可选) - 是否将梯度置零。若为 False，则删除梯度。默认值为 True。

**代码示例**

COPY-FROM: paddle.optimizer.Adadelta.clear_grad

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

COPY-FROM: paddle.optimizer.Adadelta.set_lr

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

COPY-FROM: paddle.optimizer.Adadelta.set_lr_scheduler

get_lr()
'''''''''

.. note::
    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

COPY-FROM: paddle.optimizer.Adadelta.get_lr
