.. _cn_api_paddle_optimizer_lr_OneCycleLR:

OneCycleLR
-----------------------------------

.. py:class:: paddle.optimizer.lr.OneCycleLR(max_learning_rate, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', divide_factor=25., final_divide_factor=1e4, three_phase=False, last_epoch=-1, verbose=False)

该接口使用 ``one cycle`` 策略来动态调整学习率。

该策略将学习率从初始学习率调整到最大学习率，再从最大学习率调整到远小于初始学习率的最小学习率。

相关论文： `Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates <https://arxiv.org/abs/1708.07120>`_

注意，本调度器默认行为参考fastai仓库，其声称二阶段拥有比三阶段更好的效果。设置 ``three_phase=True`` 可以与论文中所描述的行为保持一致。

同时也请注意本调度器需要在每次迭代后调用 ``step`` 方法。

参数
::::::::::::

    - **max_learning_rate** (float) - 最大学习率，学习率变化的上边界。功能上其通过 ``divide_factor`` 和 ``final_divide_factor``分别定义了初始学习率和最小学习率。
    - **total_steps** (int, optional) - 训练过程中总的迭代数。如果 ``total_steps`` 未被指定，则会根据 ``epochs`` 和 ``steps_per_epoch`` 进行计算，因此请确保 ``total_steps`` 或 ( ``epochs`` , ``steps_per_epoch`` ) 其中之一被指定。
    - **epochs** (int, optional) - 训练过程中epoch的数量。默认为 ``None`` 。
    - **steps_per_epoch** (int, optional) - 训练过程中每个epoch所需的迭代数。默认值为 ``None`` 。
    - **pct_start** (float) - 学习率从初始学习率增长到最大学习率所需迭代数占总迭代数的比例。默认值为0.3。
    - **anneal_strategy** (str, optional) - 调整学习率的策略。必须是 ( ``cos``, ``linear`` )其中之一， ``cos`` 表示使用余弦退火， ``linear`` 表示线性退火。默认值为cos。
    - **divide_factor** (float, optional) - 该参数用于推断初始学习率，公式为initial_lr = max_lr/div_factor。默认值为25。
    - **final_divide_factor** (float, optional) - 该参数用于推断最小学习率，公式为min_lr = max_lr/div_factor。默认值为25.
    - **three_phase** (bool, optional) - 是否使用三阶段调度策略。如果是 ``True`` ，学习率将先从初始学习率上升到最大学习率，再从最大学习率下降到初始学习率（这两步所需要的迭代数是一致的），最后学习率会下降至最小学习率。如果是 ``False`` ，学习率在上升至最大学习率之后，直接下降至最小学习率。默认值为 ``False`` 。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``OneCycleLR`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.OneCycleLR