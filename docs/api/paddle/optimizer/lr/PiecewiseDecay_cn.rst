.. _cn_api_paddle_optimizer_lr_PiecewiseDecay:

PiecewiseDecay
-------------------------------

.. py:class:: paddle.optimizer.lr.PiecewiseDecay(boundaries, values, last_epoch=-1, verbose=False)


该接口提供分段设置学习率的策略。`boundaries` 表示学习率变化的边界步数，对应 epoch 的值，`values` 表示学习率变化的值。

过程可以描述如下：

.. code-block:: text

    boundaries = [100, 200]  # epoch 仅代表当前步数，无实义
    values = [1.0, 0.5, 0.1] # 在第[0,100), [100,200), [200,+∞)分别对应 value 中学习率的值

    learning_rate = 1.0     if epoch < 100
    learning_rate = 0.5     if 100 <= epoch < 200
    learning_rate = 0.1     if 200 <= epoch
    ...


参数
::::::::::::

    - **boundaries** (list) - 指定学习率的边界值列表。列表的数据元素为 Python int 类型。
    - **values** (list) - 学习率列表。数据元素类型为 Python float 的列表。与边界值列表有对应的关系。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``PiecewiseDecay`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.PiecewiseDecay

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
