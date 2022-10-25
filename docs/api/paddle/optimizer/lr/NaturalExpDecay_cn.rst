.. _cn_api_paddle_optimizer_lr_NaturalExpDecay:

NaturalExpDecay
-------------------------------

.. py:class:: paddle.optimizer.lr.NaturalExpDecay(learning_rate, gama, last_epoch=-1, verbose=False)

该接口提供按自然指数衰减学习率的策略。

自然指数衰减的计算方式如下。

.. math::

    new\_learning\_rate = learning\_rate * e^{- gamma * epoch}

参数
::::::::::::

    - **learning_rate** (float) - 初始学习率，数据类型为 Python float。
    - **gamma** (float) - 衰减率，gamma 应该大于 0.0，才能使学习率衰减。默认值为 0.1。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``NaturalExpDecay`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.NaturalExpDecay

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
