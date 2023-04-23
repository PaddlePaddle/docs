.. _cn_api_paddle_optimizer_lr_OneCycleLR:

OneCycleLR
-----------------------------------

.. py:class:: paddle.optimizer.lr.OneCycleLR(max_learning_rate, total_steps, divide_factor=25., end_learning_rate=0.0001, phase_pct=0.3, anneal_strategy='cos', three_phase=False, last_epoch=-1, verbose=False)

使用 ``one cycle`` 策略来动态调整学习率。

该策略将学习率从初始学习率调整到最大学习率，再从最大学习率调整到远小于初始学习率的最小学习率。

相关论文：`Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates <https://arxiv.org/abs/1708.07120>`_

注意，本调度器默认行为参考 fastai 仓库，其声称二阶段拥有比三阶段更好的效果。设置 ``three_phase=True`` 可以与论文中所描述的行为保持一致。

同时也请注意本调度器需要在每次迭代后调用 ``step`` 方法。

参数
::::::::::::

    - **max_learning_rate** (float) - 最大学习率，学习率变化的上边界，数据类型为 Python float。功能上其通过 ``divide_factor`` 定义了初始学习率。
    - **total_steps** (int，可选) - 训练过程中总的迭代数。
    - **divide_factor** (float，可选) - 该参数用于推断初始学习率，公式为 initial_learning_rate = max_learning_rate / divide_factor。默认值为 25。
    - **end_learning_rate** (float，可选) - 最小学习率，学习率变化的下边界。它应该是一个远小于初始学习率的数。
    - **phase_pct** (float) - 学习率从初始学习率增长到最大学习率所需迭代数占总迭代数的比例。默认值为 0.3。
    - **anneal_strategy** (str，可选) - 调整学习率的策略。必须是 ( ``cos`` , ``linear`` )其中之一，``cos`` 表示使用余弦退火，``linear`` 表示线性退火。默认值为 ``cos`` 。
    - **three_phase** (bool，可选) - 是否使用三阶段调度策略。如果是 ``True``，学习率将先从初始学习率上升到最大学习率，再从最大学习率下降到初始学习率（这两阶段所需要的迭代数是一致的），最后学习率会下降至最小学习率。如果是 ``False``，学习率在上升至最大学习率之后，直接下降至最小学习率。默认值为 ``False`` 。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``OneCycleLR`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.OneCycleLR

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
