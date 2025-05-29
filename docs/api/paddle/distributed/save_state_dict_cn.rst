.. _cn_api_paddle_distributed_save_state_dict:

save_state_dict
-------------------------------

.. py:function:: paddle.distributed.save_state_dict(state_dict, path, process_group=None, coordinator_rank=0, unique_id=None, async_save=False)

保存分布式训练的 state_dict 到指定路径。

参数
:::::::::
    - **state_dict** (dict[str, paddle.Tensor]): 要保存的 state_dict。
    - **path** (str): checkpoint 文件所在目录。
    - **process_group** (paddle.distributed.collective.Group，可选): 用于跨 rank 同步的 ProcessGroup。默认值为 None，表示使用包含所有卡的全局 process group。
    - **coordinator_rank** (int，可选): 用于协调检查点的 Rank。默认值为 0，表示使用 Rank 0 作为协调检查点。
    - **unique_id** (int，可选): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None，表示使用指定路径最大值加载最新版本的检查点。
    - **async_save** (bool，可选): 是否异步保存 state_dict。默认值为 False，表示不使用异步保存。

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.save_state_dict
