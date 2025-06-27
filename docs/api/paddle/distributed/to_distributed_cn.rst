.. _cn_api_paddle_distributed_to_distributed:

to_distributed
-------------------------------

.. py:function:: paddle.distributed.to_distributed(model, optimizer, dataloader, device_num, node_num=1, config=None)

能够自动地将没有包含任何分布式代码的神经网络、优化器、数据加载器 转化为适合分布式运行的 神经网络、优化器、数据加载器 并确保正确性，同时转化过程中会根据机器数和每台机器的设备数自动选择最优的分布式策略以尽可能提升性能。


.. note::
    此接口处于原型试用阶段，支持部分模型结构在单机多卡运行，后续会扩大支持的模型范围及支持多机多卡运行。


参数
:::::::::

    - **model** (paddle.nn.Layer) - 单卡视角的模型，没有包含任何分布式代码。
    - **optimizer** (paddle.optimizer.Optimizer) - 单卡视角的优化器，通过常规优化器 API 构造，如 ``paddle.optimizer.Adam``。
    - **dataloader** (paddle.io.DataLoader) - 单卡视角的数据加载器，通过常规方式沟通，如 ``paddle.io.Dataset`` 及 ``paddle.io.Sampler``, 无需使用 ``paddle.io.DistributedBatchSampler``。
    - **config** (ToDistributedConfig，可选) - 可以用来配置 输入数据信息 和 是否使用序列并行。配置时使用数据类 ``paddle.distributed.auto_parallel.high_level_api.ToDistributedConfig`` 来完成。

      配置 输入数据信息，是提供模型训练时最有可能输入数据的 shape、dtype 和 stop_gradient 信息，便于更快更准地自动选择最优的分布式策略。

      配置 是否使用序列并行，可以指定如果最优的分布式策略中包含模型并行时，是否要使用序列并行。

返回
:::::::::
Model：一个具有分布式信息的 `paddle.nn.Layer` 对象，根据自动选择的最优分布式策略，可能包含分布式化的权重参数。

Optimizer：一个 `Optimizer` 对象，根据自动选择的最优分布式策略，可能包含分布式化的优化器状态。

DataLoader：一个 `ShardDataloader` 对象。能够给后续的分布式训练提供输入数据。


代码示例
:::::::::

COPY-FROM: paddle.distributed.to_distributed
