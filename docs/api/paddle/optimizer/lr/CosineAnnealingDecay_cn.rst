.. _cn_api_paddle_optimizer_lr_CosineAnnealingDecay:

CosineAnnealingDecay
-----------------------------------

.. py:class:: paddle.optimizer.lr.CosineAnnealingDecay(learning_rate, T_max, eta_min=0, last_epoch=-1, verbose=False)

该接口使用 ``cosine annealing`` 的策略来动态调整学习率。

.. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}


:math:`\eta_{max}` 的初始值为 ``learning_rate``， :math:`T_{cur}` 是 SGDR（重启训练 SGD）训练过程中的当前训练轮数。SGDR 的训练方法可以参考论文，
这里只是实现了 ``cosine annealing`` 动态学习率，热启训练部分没有实现。

相关论文：`SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_

参数
::::::::::::

    - **learning_rate** (float) - 初始学习率，也就是公式中的 :math:`\eta_{max}`，数据类型为 Python float或int。
    - **T_max** (float|int) - 训练的上限轮数，是余弦衰减周期的一半。必须是一个正整数。
    - **eta_min** (float|int，可选) - 学习率的最小值，即公式中的 :math:`\eta_{min}`。默认值为 0。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``CosineAnnealingDecay`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.CosineAnnealingDecay

方法
::::::::::::
step(epoch=None)
'''''''''

step 函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据 epoch 数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

**参数**

  - **epoch** （int，可选）- 指定具体的 epoch 数。默认值 None，此时将会从-1 自动累加 ``epoch`` 数。

**返回**

  无。

代码示例：
::::::::::::

  参照上述示例代码。
