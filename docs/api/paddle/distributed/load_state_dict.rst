.. _cn_api_paddle_distributed_load_state_dict:

load_state_dict
-------------------------------
将指定路径的 checkpoint 加载到指定 state_dict 中。

.. py:function:: paddle.distributed.load_state_dict(state_dict: dict[str, Tensor],
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    offload: bool = False,
    mw_name_compatibility: bool = True)


参数
:::::::::
state_dict(Dict[str, paddle.Tensor]): 要加载的 state_dict，使用原地加载方式。
path(str): checkpoint 文件所在目录。
process_group(paddle.distributed.collective.Group): 用于跨 rank 同步的 ProcessGroup。默认使用包含所有卡的全局 process group。
coordinator_rank(int): 用于协调检查点的 Rank。默认使用 Rank 0.
unique_id(int): checkpoint 的唯一 ID，用于区分不同版本的检查点。默认值为 None，使用指定路径最大值加载最新版本的检查点。
offload(bool): 是否 offload checkpoint 到 CPU。默认值为 False。
mw_name_compatibility(bool): 是否兼容动态图与静态图半自动并行参数的命名。默认值为 True。

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.distributed.load_state_dict
