    .. _cn_api_paddle_distributed_save_state_dict:

save_state_dict
-------------------------------

.. py:function:: paddle.distributed.save_state_dict(state_dict, path, process_group=None, coordinator_rank=0, unique_id=None, async_save=False, safetensors=False, save_replicas=False)

保存分布式训练的 state_dict 到指定路径。

参数
:::::::::
    - **state_dict** (dict[str, paddle.Tensor|paddle.distributed.ShardedWeight]): 要保存的 state_dict。
    - **path** (str): checkpoint 文件所在目录。
    - **process_group** (paddle.distributed.collective.Group，可选): 用于跨 rank 同步的 ProcessGroup。默认值为 None，表示使用包含所有卡的全局 process group。
    - **coordinator_rank** (int，可选): 用于协调检查点的 Rank。默认值为 0，表示使用 Rank 0 作为协调检查点。
    - **unique_id** (int，可选): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None：首次保存时使用 0，之后在同一路径中每次调用时递增 1；指定的 ID 已存在时会覆盖对应 checkpoint。
    - **async_save** (bool，可选): 是否异步保存 state_dict。默认值为 False，表示不使用异步保存。
    - **safetensors** (bool，可选): 是否使用 safetensors 格式保存。默认值为 False。
    - **save_replicas** (bool，可选): 是否保存所有 Tensor 副本（例如来自不同 rank 的副本），而非每个 Tensor 只保存一个去重后的副本。默认值为 False。

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.save_state_dict
