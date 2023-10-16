.. _cn_api_paddle_optimizer_lr_LinearLR:

LinearLR
-----------------------------------

.. py:class:: paddle.optimizer.lr.LinearLR(learning_rate, total_steps, start_factor=1./3, end_factor=1.0, last_epoch=-1, verbose=False)


该接口提供一种学习率优化策略-线性学习率对学习率进行调整。


参数
::::::::::::

    - **learning_rate** (float) - 基础学习率，用于确定初始学习率和最终学习率。
    - **total_steps** (float) - 学习率从初始学习率线性增长到最终学习率所需要的步数。
    - **start_factor** (float) - 初始学习率因子，通过 `learning_rate * start_factor` 确定。
    - **end_factor** (float) - 最终学习率因子，通过 `learning_rate * end_factor` 确定。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``LinearLR`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.LinearLR:code-dynamic
COPY-FROM: paddle.optimizer.lr.LinearLR:code-static

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
