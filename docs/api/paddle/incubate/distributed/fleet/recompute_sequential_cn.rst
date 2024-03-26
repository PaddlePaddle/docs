.. _cn_api_paddle_incubate_distributed_fleet_recompute_sequential:

recompute_sequential
-------------------------------

.. py:function:: paddle.incubate.distributed.fleet.recompute_sequential(ctx, functions, *args, **kwargs)
重新计算中间激活以节省“顺序”模型的内存。使用“ctx”来传输一些上下文参数，它类似于“recompute_hybrid”API。

参数
::::::::::::

    - **ctx** (dict) – 包含“segments”和“preserve_rng_state”键，键“segments”（int，默认 1）表示要在模型中创建的块的数量，键“preserve_rng_state”（bool，可选，默认=True）表示是否保存向前 rng。如果为 True，则在执行反向传播的正向重新计算时，将恢复最后一个正向 rng 值。
    - **function** (paddle.nn.Sequential) - 层序列的层，描述模型的前向通道的一部分，其中间激活将在前向阶段被释放以节省内存，并将在后向阶段被重新计算以进行梯度计算。
    - ***args** (Tensor) - 函数的输入（元组）。
    - ****kwargs** (Dict) - 函数的输入（字典）。

返回
:::::::::

args 和 kwargs 上的函数输出。

代码示例
::::::::::::

COPY-FROM: paddle.incubate.distributed.fleet.recompute_sequential
