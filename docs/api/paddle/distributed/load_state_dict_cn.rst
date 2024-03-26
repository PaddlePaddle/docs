.. _cn_api_paddle_distributed_load_state_dict:

load_state_dict
-------------------------------


.. py:function:: paddle.distributed.load_state_dict(state_dict, path, process_group=None, coordinator_rank=0)

用于自动并行场景下，从指定路径加载参数。

参数
:::::::::
    - **state_dict** (dict) - 待加载的参数。
    - **path** (str) - 加载参数的路径。
    - **process_group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **coordinator_rank** (int，可选) - 指定协调者的 rank。默认为 0。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.load_state_dict
