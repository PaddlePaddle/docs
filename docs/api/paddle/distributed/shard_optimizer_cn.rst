.. _cn_api_paddle_distributed_shard_optimizer:

shard_optimizer
-------------------------------

.. py:function:: paddle.distributed.shard_optimizer(optimizer, shard_fn=None)

将单卡视角的优化器转变为分布式视角。可以通过指定 `shard_fn` 来定制化优化器状态的切分方式，否则会将参数的分布式信息传递给对应的优化器状态。

`shard_fn` 的函数签名为：def shard_fn(accumulator_name, param, accumulator) -> sharded_accumulator。


参数
:::::::::

    - **optimizer** (paddle.optimizer.Optimizer) - 单卡视角的优化器。
    - **shard_fn** (Callable) - 用于切分优化器状态函数。如果没有指定，默认地我们将参数的分布式信息传递给对应的优化器状态。

返回
:::::::::
Optimizer：一个具有分布式视角的 `Optimizer` 对象。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_optimizer
