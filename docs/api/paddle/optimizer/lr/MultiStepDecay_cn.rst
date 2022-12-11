.. _cn_api_paddle_optimizer_lr_MultiStepDecay:

MultiStepDecay
-----------------------------------

.. py:class:: paddle.optimizer.lr.MultiStepDecay(learning_rate, milestones, gamma=0.1, last_epoch=-1, verbose=False)

该接口提供一种学习率按 **指定轮数** 进行衰减的策略。

衰减过程可以参考以下代码：

.. code-block:: text

    learning_rate = 0.5
    milestones = [30, 50]
    gamma = 0.1

    learning_rate = 0.5     if epoch < 30
    learning_rate = 0.05    if 30 <= epoch < 50
    learning_rate = 0.005   if 50 <= epoch
    ...

参数
::::::::::::

    - **learning_rate** (float) - 初始学习率，数据类型为 Python float。
    - **milestones** (list) - 轮数下标列表。必须递增。
    - **gamma** (float，可选) - 衰减率，``new_lr = origin_lr * gamma``，衰减率必须小于等于 1.0，默认值为 0.1。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。


返回
::::::::::::
用于调整学习率的 ``MultiStepDecay`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.MultiStepDecay

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
