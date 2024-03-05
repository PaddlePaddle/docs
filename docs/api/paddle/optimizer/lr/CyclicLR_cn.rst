.. _cn_api_paddle_optimizer_lr_CyclicLR:

CyclicLR
-----------------------------------

.. py:class:: paddle.optimizer.lr.CyclicLR(base_learning_rate, max_learning_rate, step_size_up, step_size_down=None, mode='triangular', exp_gamma=1., scale_fn=None, scale_mode='cycle', last_epoch=-1, verbose=False)

提供一种学习率按固定频率在两个边界之间循环的策略。

该策略将学习率调整的过程视为一个又一个的循环，学习率根据指定的缩放策略以固定的频率在最大和最小学习率之间变化。

相关论文：`Cyclic Learning Rates for Training Neural Networks <https://arxiv.org/abs/1506.01186>`_

内置了三种学习率缩放策略：**triangular**：没有任何缩放的三角循环。**triangular2**：每个三角循环里将初始幅度缩放一半。**exp_range**：每个循环中将初始幅度按照指数函数进行缩放，公式为 :math:`gamma^{iterations}`。

初始幅度由 `max_learning_rate - base_learning_rate` 定义。同时需要注意 CyclicLR 应在每次迭代后调用 ``step`` 方法。

参数
::::::::::::

    - **base_learning_rate** (float) - 初始学习率，也是学习率变化的下边界。论文中建议将其设置为最大学习率的三分之一或四分之一。
    - **max_learning_rate** (float) - 最大学习率，需要注意的是，实际的学习率由 ``base_learning_rate`` 与初始幅度的缩放求和而来，因此实际学习率可能达不到 ``max_learning_rate`` 。
    - **step_size_up** (int) - 学习率从初始学习率增长到最大学习率所需步数。每个循环总的步长 ``step_size`` 由 ``step_size_up + step_size_down`` 定义，论文中建议将 ``step_size`` 设置为单个 epoch 中步长的 3 或 4 倍。
    - **step_size_down** (int，可选) - 学习率从最大学习率下降到初始学习率所需步数。若未指定，则其值默认等于 ``step_size_up`` 。
    - **mode** (str，可选) - 可以是 triangular、triangular2 或者 exp_range，对应策略已在上文描述，当 scale_fn 被指定时时，该参数将被忽略。默认值为 triangular。
    - **exp_gamma** (float，可选) - exp_range 缩放函数中的常量。默认值为 1.0。
    - **sacle_fn** (function，可选) - 一个有且仅有单个参数的函数，且对于任意的输入 x，都必须满足 0 ≤ scale_fn(x) ≤ 1；如果该参数被指定，则会忽略 mode 参数。默认值为 ``False`` 。
    - **scale_mode** (str，可选) - cycle 或者 iterations，表示缩放函数使用 cycle 数或 iterations 数作为输入。默认值为 cycle。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``CyclicLR`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.CyclicLR

方法
::::::::::::
step(epoch=None)
'''''''''

step 函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据 epoch 数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

**参数**

  - **epoch** (int，可选) - 指定具体的 epoch 数。默认值 None，此时将会从-1 自动累加 ``epoch`` 数。

**返回**

无。

**代码示例**

参照上述示例代码。
