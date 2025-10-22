.. _cn_api_paddle_distributed_load_state_dict:

load_state_dict
-------------------------------
.. py:function:: paddle.distributed.load_state_dict(state_dict, path, process_group=None, coordinator_rank=0, unique_id=None, offload=False, mw_name_compatibility=True)

将指定路径的 checkpoint 加载到指定 state_dict 中。


参数
:::::::::
    - **state_dict** (dict[str, paddle.Tensor]): 要加载的 state_dict，使用原地加载方式。
    - **path** (str): checkpoint 文件所在目录。
    - **process_group** (paddle.distributed.collective.Group，可选): 用于跨 rank 同步的 ProcessGroup。默认值为 None，表示使用包含所有卡的全局 process group。
    - **coordinator_rank** (int，可选): 用于协调检查点的 Rank。默认值为 0，表示使用 Rank 0 作为协调检查点。
    - **unique_id** (int，可选): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None，使用指定路径最大值加载最新版本的检查点。
    - **offload** (bool，可选): 是否 offload checkpoint 到 CPU。默认值为 False，表示不进行 offload。
    - **mw_name_compatibility** (bool，可选): 是否兼容动态图与静态图半自动并行参数的命名。默认值为 True，表示兼容。

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.load_state_dict
