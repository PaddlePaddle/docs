.. _cn_api_paddle_distributed_init_parallel_env:

init_parallel_env
-----------------

.. py:function:: paddle.distributed.init_parallel_env(nccl_config=None)

初始化动态图模式下的并行训练环境。

.. note::
    目前同时初始化 ``NCCL`` 和 ``GLOO`` 上下文用于通信。

参数
:::::::::

    - **nccl_config** (NCCLConfig|None，可选) - NCCL 配置。默认值为 None。

返回
:::::::::
Group，初始化后的默认进程组实例

代码示例
:::::::::
COPY-FROM: paddle.distributed.init_parallel_env
