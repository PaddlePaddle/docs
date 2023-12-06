.. _cn_api_paddle_optimizer_Lamb:

Lamb
-------------------------------

.. py:class:: paddle.optimizer.Lamb(learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, parameters=None, grad_clip=None, exclude_from_weight_decay_fn=None, name=None)



LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器旨在不降低精度的前提下增大训练的批量大小，其支持自适应的逐元素更新和精确的分层校正。参数更新如下：

.. math::
    m_t=\beta_1*m_{t-1} + (1-\beta_1)*g_t
.. math::
    v_t=\beta_2∗v_{t-1}+(1−\beta_2)∗g_t^2
.. math::
    m_t=\frac{m_t}{1-\beta_1^t}
.. math::
    v_t=\frac{v_t}{1-\beta_2^t}
.. math::
    r_t=\frac{m_t}{\sqrt{v_t}+\epsilon}
.. math::
    w_t=w_{t_1}-\eta_t*\frac{\left \| w_{t-1}\right \|}{\left \| r_t+\lambda*w_{t-1}\right \|}*(r_t+\lambda*w_{t-1}) \\

其中 :math:`m` 表示第一个动量，:math:`v` 代表第二个动量，:math:`\eta` 代表学习率，:math:`\lambda` 代表 LAMB 的权重学习率。

相关论文：`Large Batch Optimization for Deep Learning: Training BERT in 76 minutes <https://arxiv.org/pdf/1904.00962.pdf>`_

参数
::::::::::::

  - **learning_rate** (float|Tensor，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个 Tensor，默认值为 0.001。
  - **lamb_weight_decay** (float，可选) – LAMB 权重衰减率。默认值为 0.01。
  - **beta1** (float，可选) - 第一个动量估计的指数衰减率。默认值为 0.9。
  - **beta2** (float，可选) - 第二个动量估计的指数衰减率。默认值为 0.999。
  - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为 1e-06。
  - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
  - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>`。默认值为 None，此时将不进行梯度裁剪。
  - **exclude_from_weight_decay_fn** (function) - 当某个参数作为输入该函数返回值为 True 时，为该参数跳过权重衰减。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

.. note::
    目前 ``Lamb`` 不支持 Sparse Parameter Optimization（稀疏参数优化）。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.Lamb

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

COPY-FROM: paddle.optimizer.Lamb.step

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters 中的 Parameters，最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量。
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_paddle_static_Program`，默认值为 None，此时将使用 :ref:`cn_api_paddle_static_default_startup_program`。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成集合，默认值为 None。

**返回**

 tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

**代码示例**

COPY-FROM: paddle.optimizer.Lamb.minimize

clear_grad()
'''''''''

.. note::
该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

COPY-FROM: paddle.optimizer.Lamb.clear_grad

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

COPY-FROM: paddle.optimizer.Lamb.set_lr

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
    lamb = paddle.optimizer.Lamb(0.1, parameters=linear.parameters())
    # set learning rate manually by class LRScheduler
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2,4,6], gamma=0.8)
    lamb.set_lr_scheduler(scheduler)
    lr = lamb.get_lr()
    print("current lr is {}".format(lr))
    #    current lr is 0.5
    # set learning rate manually by another LRScheduler
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=5, gamma=0.6)
    lamb.set_lr_scheduler(scheduler)
    lr = lamb.get_lr()
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

COPY-FROM: paddle.optimizer.Lamb.get_lr
