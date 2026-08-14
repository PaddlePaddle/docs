.. _cn_api_paddle_distributed_load_state_dict:

load_state_dict
-------------------------------
.. py:function:: paddle.distributed.load_state_dict(state_dict, path, process_group=None, coordinator_rank=0, unique_id=None, offload=False, mw_name_compatibility=True, aoa_config=None, safetensors=False, worker_groups=None, comm_method="broadcast")

将指定路径的 checkpoint 加载到指定 state_dict 中。


参数
:::::::::
    - **state_dict** (dict[str, paddle.Tensor|paddle.distributed.ShardedWeight]): 要加载的 state_dict，使用原地加载方式。
    - **path** (str): checkpoint 文件所在目录。
    - **process_group** (paddle.distributed.collective.Group，可选): 用于跨 rank 同步的 ProcessGroup。默认值为 None，表示使用包含所有卡的全局 process group。
    - **coordinator_rank** (int，可选): 用于协调检查点的 Rank。默认值为 0，表示使用 Rank 0 作为协调检查点。
    - **unique_id** (int，可选): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None，使用指定路径最大值加载最新版本的检查点。
    - **offload** (bool，可选): 是否 offload checkpoint 到 CPU。默认值为 False，表示不进行 offload。
    - **mw_name_compatibility** (bool，可选): 是否兼容动态图与静态图半自动并行参数的命名。默认值为 True，表示兼容。
    - **aoa_config** (dict[str, list[str]]|None，可选): 用于修改参数的 AOA 配置。默认值为 None。
    - **safetensors** (bool，可选): 是否使用 safetensors 格式。默认值为 False。
    - **worker_groups** (list[paddle.distributed.collective.Group]|None，可选): 用于 Tensor 通信的通信组。提供多个通信组时会选择合适的组；为 None 时使用 process_group。默认值为 None。
    - **comm_method** (str，可选): 重分片的通信方式，可选 ``"send_recv"``、``"broadcast"``、``"multi_group_broadcast"``、``"grouped_send_recv"`` 或 ``"parallel_broadcast"``。默认值为 ``"broadcast"``。

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.load_state_dict
