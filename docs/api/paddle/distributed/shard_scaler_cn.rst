.. _cn_api_paddle_distributed_shard_scaler:

shard_scaler
-------------------------------

.. py:function:: paddle.distributed.shard_scaler(scaler)

将单卡视角的 GradScaler 转变为分布式视角。


参数
:::::::::

    - **scaler** (paddle.amp.GradScaler) - 单卡视角的 `GradScaler`。
返回
:::::::::
GradScaler：一个具有分布式视角的 `GradScaler` 对象。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_scaler
