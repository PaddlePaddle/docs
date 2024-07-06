.. _cn_api_paddle_optimizer_NAdam:

NAdam
-------------------------------

.. py:class:: paddle.optimizer.NAdam(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1.0e-8, momentum_decay=0.004, parameters=None, weight_decay=None, grad_clip=None, name=None)


在论文 `Incorporating Nesterov Momentum into Adam <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>`_ 中， NAdam 优化器的实现是基于 :ref:`Adam <cn_api_paddle_optimizer_Adam>` 优化算法实现的。主要的改进是结合了 Nesterov 动量和 Adam 自适应学习率的优点。

其参数更新的计算公式如下：

.. math::

    \begin{aligned}
        &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
        &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  \rho ^{t \psi} \big)     \\
        &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96 ^{(t+1)\psi}\big)\\
        &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
        &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
        &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i) + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
        &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
        &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
            \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
        &\hspace{0mm} \text{ with: } \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)}, \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)} \\
        &\hspace{0mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)} \\
    \end{aligned}


参数
::::::::::::

  - **learning_rate** (float|LRScheduler，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个 `LRScheduler` 类。默认值为 0.002。
  - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
  - **beta1** (float，可选) - 一阶矩估计的指数衰减率，默认值为 0.9。
  - **beta2** (float，可选) - 二阶矩估计的指数衰减率，默认值为 0.999。
  - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为 1e-08。
  - **weight_decay** (float|Tensor，可选) - 正则化方法。可以是 float 类型或者 Tensor。默认值为 None，表示没有正则化。
  - **momentum_decay** (float，可选): 动量衰减率。默认值为 0.004.
  - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。默认值为 None，此时将不进行梯度裁剪。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

.. note::

    目前 ``NAdam`` 不支持 Sparse Parameter Optimization（稀疏参数优化）。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.NAdam

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

COPY-FROM: paddle.optimizer.NAdam.step

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

COPY-FROM: paddle.optimizer.NAdam.minimize


clear_grad()
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

COPY-FROM: paddle.optimizer.NAdam.clear_grad

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

COPY-FROM: paddle.optimizer.NAdam.set_lr

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

COPY-FROM: paddle.optimizer.NAdam.set_lr_scheduler

get_lr()
'''''''''

.. note::

该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

COPY-FROM: paddle.optimizer.NAdam.get_lr
