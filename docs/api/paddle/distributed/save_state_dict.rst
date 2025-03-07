.. _cn_api_paddle_distributed_save_state_dict:

save_state_dict
-------------------------------
保存分布式训练的 state_dict 到指定路径。

.. py:function:: paddle.distributed.save_state_dict(state_dict: dict[str, Tensor],
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    async_save: bool = False)


参数
:::::::::
state_dict(Dict[str, paddle.Tensor]): 要保存的 state_dict。
path(str): checkpoint 文件所在目录。
process_group(paddle.distributed.collective.Group): 用于跨 rank 同步的 ProcessGroup。默认使用包含所有卡的全局 process group。
coordinator_rank(int): 用于协调检查点的 Rank。默认使用 Rank 0.
unique_id(int): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None，使用指定路径最大值加载最新版本的检查点。
async_save(bool): 异步保存 state_dict。默认值为 False。


返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.save_state_dict
