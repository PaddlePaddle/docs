.. _cn_api_paddle_distributed_load_state_dict:

load_state_dict
-------------------------------


.. py:function:: paddle.distributed.load_state_dict(state_dict, path, process_group=None, coordinator_rank=0)

从指定的checkpoint路径原地加载state_dict。

参数
:::::::::
    - **state_dict** (Dict[str, paddle.Tensor]) - 要保存的state_dict。
    - **path** (str) - 保存state_dict的目录。
    - **process_group** (paddle.distributed.collective.Group，可选) - 用于跨rank同步的ProcessGroup。默认为 None，即使用全局默认进程组。
    - **coordinator_rank** (int，可选) - 用于保存非分布式值的rank。默认使用Rank0。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.load_state_dict
