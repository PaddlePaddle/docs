.. _cn_api_paddle_optimizer_lr_NoamDecay:

NoamDecay
-------------------------------

.. py:class:: paddle.optimizer.lr.NoamDecay(d_model, warmup_steps, learning_rate=1.0, last_epoch=-1, verbose=False)


该接口提供 Noam 衰减学习率的策略。

Noam 衰减的计算方式如下：

.. math::

    new\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(epoch^{-0.5}, epoch * warmup\_steps^{-1.5})

相关论文：`attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 。

参数
::::::::::::

    - **d$_{model}$** (int) - 模型的输入、输出向量特征维度，为超参数。数据类型为 Python int。
    - **warmup_steps** (int) - 预热步数，为超参数。数据类型为 Python int。
    - **learning_rate** (float) - 初始学习率，数据类型为 Python float。默认值为 1.0。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 `True`，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``NoamDecay`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.NoamDecay

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
