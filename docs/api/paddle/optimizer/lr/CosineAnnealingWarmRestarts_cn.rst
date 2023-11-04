.. _cn_api_paddle_optimizer_lr_CosineAnnealingWarmRestarts:

CosineAnnealingWarmRestarts
-----------------------------------

.. py:class:: paddle.optimizer.lr.CosineAnnealingWarmRestarts(learning_rate, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)

该接口使用 ``cosine annealing`` 的策略来动态调整学习率。

.. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

:math:`\eta_{max}` 的初始值为 ``learning_rate``， :math:`T_{cur}` 是 SGDR（重启训练 SGD）训练过程中的当前训练轮数。:math:`T_{i}` 是 SGDR 两次重启训练之间epoch的数量

当 :math:`T_{cur}=T_{i}` ，设 :math:`\eta_t = \eta_{min}` 。当重启后 :math:`T_{cur}=0` ，设 :math:`\eta_t=\eta_{max}` 。

SGDR 的训练方法可以参考论文，相关论文：`SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_

参数
::::::::::::

    - **learning_rate** (float) - 初始学习率。
    - **T_0** (int) - 首次重启后迭代的次数。
    - **T_mult** (int，可选) - 重启之后 :math:`T_{i}` 乘积增长因子。默认值 1。
    - **eta_min** (float，可选) - 最小学习率。默认值 0.
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``CosineAnnealingWarmRestarts`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.CosineAnnealingWarmRestarts:code-example1
COPY-FROM: paddle.optimizer.lr.CosineAnnealingWarmRestarts:code-example2

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
