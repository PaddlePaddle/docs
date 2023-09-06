.. _cn_api_paddle_optimizer_lr_LinearWarmup:

LinearWarmup
-----------------------------------

.. py:class:: paddle.optimizer.lr.LinearWarmup(learning_rate, warmup_steps, start_lr, end_lr, last_epoch=-1, verbose=False)

该接口提供一种学习率优化策略-线性学习率热身(warm up)对学习率进行初步调整。在正常调整学习率之前，先逐步增大学习率。

当训练步数小于热身步数（warmup_steps）时，学习率 lr 按如下方式更新：

.. math::

    lr = start\_lr + (end\_lr - start\_lr) * \frac{epoch}{warmup\_steps}

当训练步数大于等于热身步数（warmup_steps）时，学习率 lr 为：

.. math::

    lr = learning\_rate

其中 learning_rate 为热身之后的学习率，可以是 python 的 float 类型或者 ``_LRScheduler`` 的任意子类。

参数
::::::::::::

    - **learning rate** (float|_LRScheduler) - 热启训练之后的学习率，可以是 python 的 float 类型或者 ``_LRScheduler`` 的任意子类。
    - **warmup_steps** (int) - 进行 warm up 过程的步数。
    - **start_lr** (float) - warm up 的起始学习率。
    - **end_lr** (float) - warm up 的最终学习率。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。


返回
::::::::::::
用于调整学习率的 ``LinearWarmup`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.LinearWarmup

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
