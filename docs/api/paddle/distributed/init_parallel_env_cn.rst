.. _cn_api_paddle_distributed_init_parallel_env:

init_parallel_env
-----------------

.. py:function:: paddle.distributed.init_parallel_env()

初始化动态图模式下的并行训练环境。

.. note::
    目前同时初始化 ``NCCL`` 和 ``GLOO`` 上下文用于通信。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.init_parallel_env
